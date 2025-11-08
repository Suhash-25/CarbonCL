"""FastAPI application for the Compute Metering Service."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from .intensity import router as intensity_router
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
# Include carbon intensity router
app.include_router(intensity_router)

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