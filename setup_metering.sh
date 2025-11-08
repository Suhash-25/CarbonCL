#!/bin/bash
# Complete setup script for Compute Metering Service

echo "Creating Compute Metering Service structure..."

# Create app directory
mkdir -p app
mkdir -p data

# Create __init__.py
cat > app/__init__.py << 'ENDOFINIT'
"""Compute Metering Service - Energy and CO₂ measurement for shell commands."""

__version__ = "1.0.0"
ENDOFINIT

# Create settings.py
cat > app/settings.py << 'ENDOFSETTINGS'
"""Configuration settings for the Compute Metering Service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    MEASURE_POWER_SECS: int = 1
    DATA_DIR: str = "./data"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
ENDOFSETTINGS

# Create models.py
cat > app/models.py << 'ENDOFMODELS'
"""Data models for the Compute Metering Service."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RunStatus(str, Enum):
    """Status of a metered run."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELED = "canceled"


class StartRunRequest(BaseModel):
    """Request to start a new metered run."""
    
    command: list[str] = Field(..., min_length=1, description="Shell command as array of arguments")
    env: Optional[dict[str, str]] = Field(None, description="Environment variables to merge")
    timeout_seconds: Optional[int] = Field(None, ge=1, description="Timeout in seconds")
    tags: Optional[dict[str, str]] = Field(None, description="Custom tags for the run")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "command": ["sleep", "5"],
                    "env": {"MY_VAR": "test"},
                    "timeout_seconds": 30,
                    "tags": {"team": "backend", "service": "api"}
                }
            ]
        }
    }


class RunInfo(BaseModel):
    """Information about a metered run."""
    
    run_id: str = Field(..., description="Unique identifier for the run")
    status: RunStatus = Field(..., description="Current status of the run")
    started_at: Optional[datetime] = Field(None, description="When the run started")
    ended_at: Optional[datetime] = Field(None, description="When the run ended")
    seconds_elapsed: float = Field(0.0, ge=0, description="Total elapsed time in seconds")
    energy_kwh: float = Field(0.0, ge=0, description="Total energy consumed in kWh")
    emissions_kg: float = Field(0.0, ge=0, description="Total CO2 emissions in kg")
    cpu_energy_kwh: Optional[float] = Field(None, ge=0, description="CPU energy in kWh")
    gpu_energy_kwh: Optional[float] = Field(None, ge=0, description="GPU energy in kWh")
    notes: Optional[str] = Field(None, description="Additional notes or error messages")
    exit_code: Optional[int] = Field(None, description="Process exit code")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "run_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "done",
                    "started_at": "2025-01-15T10:30:00Z",
                    "ended_at": "2025-01-15T10:30:05Z",
                    "seconds_elapsed": 5.2,
                    "energy_kwh": 0.00015,
                    "emissions_kg": 0.000075,
                    "cpu_energy_kwh": 0.00015,
                    "gpu_energy_kwh": None,
                    "notes": None,
                    "exit_code": 0
                }
            ]
        }
    }


class StartRunResponse(BaseModel):
    """Response after starting a run."""
    
    run_id: str = Field(..., description="Unique identifier for the started run")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"run_id": "550e8400-e29b-41d4-a716-446655440000"}
            ]
        }
    }


class CancelRunResponse(BaseModel):
    """Response after canceling a run."""
    
    status: str = Field(..., description="Cancellation status")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "canceled"}
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service health status")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "ok"}
            ]
        }
    }
ENDOFMODELS

# Create tracker.py
cat > app/tracker.py << 'ENDOFTRACKER'
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
            
            # Initialize CodeCarbon tracker
            tracker = EmissionsTracker(
                measure_power_secs=settings.MEASURE_POWER_SECS,
                tracking_mode="process",
                output_dir=settings.DATA_DIR,
                save_to_file=True,
                log_level=settings.LOG_LEVEL,
                project_name=f"run_{run_id}"
            )
            
            with self._lock:
                self._trackers[run_id] = tracker
            
            tracker.start()
            
            # Prepare environment
            env = os.environ.copy()
            if req.env:
                env.update(req.env)
            
            # Start the process
            process = subprocess.Popen(
                req.command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
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
ENDOFTRACKER

# Create main.py
cat > app/main.py << 'ENDOFMAIN'
"""FastAPI application for the Compute Metering Service."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from .models import (
    CancelRunResponse,
    HealthResponse,
    RunInfo,
    StartRunRequest,
    StartRunResponse,
)
from .settings import settings
from .tracker import RunManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global run manager instance
run_manager: Optional[RunManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    global run_manager
    
    # Startup
    logger.info("Starting Compute Metering Service")
    run_manager = RunManager()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Compute Metering Service")


app = FastAPI(
    title="Compute Metering Service",
    description="Measure energy and CO₂ emissions of shell commands using CodeCarbon",
    version="1.0.0",
    lifespan=lifespan
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is running"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        Health status response
    """
    return HealthResponse(status="ok")


@app.post(
    "/runs",
    response_model=StartRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a metered run",
    description="Start a new metered run for the given command"
)
async def start_run(request: StartRunRequest) -> StartRunResponse:
    """
    Start a new metered run.
    
    Args:
        request: The run configuration
        
    Returns:
        Response with the run_id
        
    Raises:
        HTTPException: If the run cannot be started
    """
    if run_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    try:
        run_id = run_manager.start_run(request)
        return StartRunResponse(run_id=run_id)
    except Exception as e:
        logger.exception("Failed to start run")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start run: {str(e)}"
        )


@app.get(
    "/runs/{run_id}",
    response_model=RunInfo,
    summary="Get run information",
    description="Get current status and metrics for a run"
)
async def get_run(run_id: str) -> RunInfo:
    """
    Get information about a run.
    
    Args:
        run_id: The unique run identifier
        
    Returns:
        Current run information
        
    Raises:
        HTTPException: If the run is not found
    """
    if run_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    run_info = run_manager.get_info(run_id)
    
    if run_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    
    return run_info


@app.post(
    "/runs/{run_id}/cancel",
    response_model=CancelRunResponse,
    summary="Cancel a run",
    description="Cancel a running or queued job"
)
async def cancel_run(run_id: str) -> CancelRunResponse:
    """
    Cancel a run.
    
    Args:
        run_id: The unique run identifier
        
    Returns:
        Cancellation status
        
    Raises:
        HTTPException: If the run cannot be canceled
    """
    if run_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    try:
        run_manager.cancel(run_id)
        return CancelRunResponse(status="canceled")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Failed to cancel run")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel run: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )
ENDOFMAIN

echo ""
echo "✅ All files created successfully!"
echo ""
echo "File structure:"
tree app/ 2>/dev/null || find app/ -type f
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Run the service: python -m app.main"
echo "3. Test: curl http://localhost:8000/health"