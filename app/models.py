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