"""Run tracking and metering logic using CodeCarbon."""

import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Optional

from codecarbon import EmissionsTracker

from .models import RunInfo, RunStatus, StartRunRequest
from .settings import settings

logger = logging.getLogger(__name__)


class RunManager:
    """Manages metered runs with CodeCarbon integration."""
    
    def __init__(self):
        """Initialize the run manager."""
        self._runs: Dict[str, RunInfo] = {}
        self._lock = threading.Lock()
        self._processes: Dict[str, subprocess.Popen] = {}
        self._trackers: Dict[str, EmissionsTracker] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        
        # Ensure data directory exists
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        
        logger.info("RunManager initialized with DATA_DIR=%s", settings.DATA_DIR)
    
    def start_run(self, req: StartRunRequest) -> str:
        """
        Start a new metered run.
        
        Args:
            req: The run request configuration
            
        Returns:
            The unique run_id
        """
        run_id = str(uuid.uuid4())
        
        with self._lock:
            self._runs[run_id] = RunInfo(
                run_id=run_id,
                status=RunStatus.QUEUED,
                started_at=None,
                ended_at=None,
                seconds_elapsed=0.0,
                energy_kwh=0.0,
                emissions_kg=0.0,
            )
            self._cancel_flags[run_id] = threading.Event()
        
        # Start the run in a background thread
        thread = threading.Thread(
            target=self._run_worker,
            args=(run_id, req),
            daemon=True
        )
        thread.start()
        
        logger.info("Started run %s with command: %s", run_id, req.command)
        return run_id
    
    def get_info(self, run_id: str) -> Optional[RunInfo]:
        """
        Get information about a run.
        
        Args:
            run_id: The unique run identifier
            
        Returns:
            RunInfo if found, None otherwise
        """
        with self._lock:
            return self._runs.get(run_id)
    
    def cancel(self, run_id: str) -> None:
        """
        Cancel a running job.
        
        Args:
            run_id: The unique run identifier
            
        Raises:
            ValueError: If run not found or not cancelable
        """
        with self._lock:
            if run_id not in self._runs:
                raise ValueError(f"Run {run_id} not found")
            
            run_info = self._runs[run_id]
            if run_info.status not in (RunStatus.QUEUED, RunStatus.RUNNING):
                raise ValueError(f"Run {run_id} is not cancelable (status: {run_info.status})")
            
            # Set cancel flag
            if run_id in self._cancel_flags:
                self._cancel_flags[run_id].set()
        
        logger.info("Cancel requested for run %s", run_id)
    
    def _run_worker(self, run_id: str, req: StartRunRequest) -> None:
        """
        Worker thread that executes and meters a command.
        
        Args:
            run_id: The unique run identifier
            req: The run request configuration
        """
        tracker: Optional[EmissionsTracker] = None
        process: Optional[subprocess.Popen] = None
        start_time = time.time()
        
        try:
            # Update status to running
            with self._lock:
                self._runs[run_id].status = RunStatus.RUNNING
                self._runs[run_id].started_at = datetime.utcnow()
            
            # Check GPU availability
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_available = True
                logger.info("NVIDIA GPU detected and initialized")
            except Exception as e:
                gpu_available = False
                logger.warning("No NVIDIA GPU available: %s", e)
            
            # Initialize CodeCarbon tracker with GPU support if available
            tracker = EmissionsTracker(
                measure_power_secs=settings.MEASURE_POWER_SECS,
                tracking_mode="process",
                output_dir=settings.DATA_DIR,
                save_to_file=True,
                log_level=settings.LOG_LEVEL,
                project_name=f"run_{run_id}",
                gpu_ids=None if gpu_available else [],  # Track all GPUs if available
                save_to_api=False,  # Don't send to API
                logging_logger=logger  # Use our logger
            )
            
            with self._lock:
                self._trackers[run_id] = tracker
            
            tracker.start()
            
            # Prepare environment
            env = os.environ.copy()
            if req.env:
                env.update(req.env)
            
            # Start the process
            try:
                # Log detailed command information
                logger.info(
                    "Executing command for run %s: command=%s, cwd=%s",
                    run_id,
                    req.command,
                    os.getcwd()
                )
                
                # On Windows, we need to handle command resolution differently
                if os.name == 'nt':
                    # Try to resolve the command if it's a known Windows command
                    cmd_name = req.command[0].lower()
                    if cmd_name in ('python', 'python3', 'pip', 'pip3'):
                        # For Python commands, use sys.executable
                        import sys
                        cmd = [sys.executable] + req.command[1:]
                    elif not os.path.isfile(cmd_name) and '.' not in cmd_name:
                        # Add .exe extension for Windows commands if not specified
                        cmd = [cmd_name + '.exe'] + req.command[1:]
                    else:
                        cmd = req.command
                else:
                    cmd = req.command
                
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    cwd=os.getcwd()
                )
            except FileNotFoundError as e:
                # Make the error message more helpful
                cmd_name = req.command[0] if req.command else "unknown"
                error_msg = (
                    f"Command not found: '{cmd_name}'\n"
                    f"Full command: {req.command}\n"
                    f"Working directory: {os.getcwd()}\n"
                    f"PATH: {env.get('PATH', '')}\n"
                    "Please ensure the command exists and is in your PATH or provide the full path to the executable."
                )
                logger.error("Command execution failed: %s", error_msg)
                raise RuntimeError(error_msg) from e
            
            with self._lock:
                self._processes[run_id] = process
            
            # Monitor the process
            timeout = req.timeout_seconds
            poll_interval = 1.0
            elapsed = 0.0
            
            while True:
                # Check for cancellation
                if self._cancel_flags[run_id].is_set():
                    logger.info("Canceling run %s", run_id)
                    self._terminate_process(process)
                    with self._lock:
                        self._runs[run_id].status = RunStatus.CANCELED
                        self._runs[run_id].notes = "Canceled by user"
                    break
                
                # Check for timeout
                if timeout and elapsed >= timeout:
                    logger.warning("Run %s timed out after %s seconds", run_id, timeout)
                    self._terminate_process(process)
                    with self._lock:
                        self._runs[run_id].status = RunStatus.ERROR
                        self._runs[run_id].notes = f"Timeout after {timeout} seconds"
                    break
                
                # Check if process finished
                exit_code = process.poll()
                if exit_code is not None:
                    logger.info("Run %s completed with exit code %s", run_id, exit_code)
                    with self._lock:
                        self._runs[run_id].status = RunStatus.DONE
                        self._runs[run_id].exit_code = exit_code
                        if exit_code != 0:
                            stderr = process.stderr.read().decode('utf-8', errors='ignore')
                            self._runs[run_id].notes = f"Non-zero exit code: {exit_code}"
                            if stderr:
                                self._runs[run_id].notes += f"\nStderr: {stderr[:500]}"
                    break
                
                # Update elapsed time
                elapsed = time.time() - start_time
                with self._lock:
                    self._runs[run_id].seconds_elapsed = elapsed
                
                time.sleep(poll_interval)
            
            # Stop the tracker and get final metrics
            emissions_data = tracker.stop()
            
            # Extract metrics from tracker
            final_emissions_kg = 0.0
            energy_kwh = 0.0
            cpu_energy_kwh = None
            gpu_energy_kwh = None
            
            if emissions_data is not None:
                final_emissions_kg = float(emissions_data)
                
            # Try to get detailed GPU metrics
            try:
                if hasattr(tracker, '_total_gpu_energy'):
                    gpu_energy = tracker._total_gpu_energy
                    if gpu_energy and hasattr(gpu_energy, 'kWh'):
                        gpu_energy_kwh = float(gpu_energy.kWh)
                        logger.info("GPU energy consumption: %.6f kWh", gpu_energy_kwh)
                if gpu_energy_kwh is None and 'gpu_available' in locals() and gpu_available:
                    # Fallback to NVML directly
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                        duration_hours = tracker._duration_baseline / 3600  # Convert seconds to hours
                        gpu_energy_kwh = (power_watts * duration_hours) / 1000  # Convert W*h to kWh
                        logger.info("GPU %d power usage: %.2f W, duration: %.2f hours, energy: %.6f kWh", 
                                  i, power_watts, duration_hours, gpu_energy_kwh)
            except Exception as e:
                logger.warning("Could not get detailed GPU metrics: %s", e)
            
            # Try to get detailed metrics from tracker's final data
            try:
                if hasattr(tracker, '_total_energy'):
                    energy_kwh = tracker._total_energy.kWh if hasattr(tracker._total_energy, 'kWh') else 0.0
                if hasattr(tracker, '_total_cpu_energy'):
                    cpu_energy_kwh = tracker._total_cpu_energy.kWh if hasattr(tracker._total_cpu_energy, 'kWh') else None
                if hasattr(tracker, '_total_gpu_energy'):
                    gpu_energy_kwh = tracker._total_gpu_energy.kWh if hasattr(tracker._total_gpu_energy, 'kWh') else None
            except Exception as e:
                logger.warning("Could not extract detailed energy metrics: %s", e)
            
            # If we couldn't get energy from tracker internals, estimate from emissions
            if energy_kwh == 0.0 and final_emissions_kg > 0.0:
                # Rough estimate: assuming ~500g CO2/kWh average
                energy_kwh = final_emissions_kg / 0.5
            
            # Update final metrics
            with self._lock:
                self._runs[run_id].ended_at = datetime.utcnow()
                self._runs[run_id].seconds_elapsed = time.time() - start_time
                self._runs[run_id].emissions_kg = final_emissions_kg
                self._runs[run_id].energy_kwh = energy_kwh
                self._runs[run_id].cpu_energy_kwh = cpu_energy_kwh
                self._runs[run_id].gpu_energy_kwh = gpu_energy_kwh
            
            logger.info(
                "Run %s finished: status=%s, energy=%.6f kWh, emissions=%.6f kg",
                run_id,
                self._runs[run_id].status,
                energy_kwh,
                final_emissions_kg
            )
            
        except Exception as e:
            logger.exception("Error in run worker for %s", run_id)
            with self._lock:
                self._runs[run_id].status = RunStatus.ERROR
                self._runs[run_id].ended_at = datetime.utcnow()
                self._runs[run_id].seconds_elapsed = time.time() - start_time
                self._runs[run_id].notes = f"Internal error: {str(e)}"
        
        finally:
            # Cleanup
            if tracker:
                try:
                    tracker.stop()
                except Exception:
                    pass
            
            with self._lock:
                self._processes.pop(run_id, None)
                self._trackers.pop(run_id, None)
    
    def _terminate_process(self, process: subprocess.Popen, timeout: int = 5) -> None:
        """
        Terminate a process gracefully, then forcefully if needed.
        
        Args:
            process: The process to terminate
            timeout: Seconds to wait before SIGKILL
        """
        try:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate gracefully, sending SIGKILL")
                process.kill()
                process.wait()
        except Exception as e:
            logger.error("Error terminating process: %s", e)