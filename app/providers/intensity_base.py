"""Base interface and models for carbon intensity providers."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, field_validator


class IntensityPoint(BaseModel):
    """A single carbon intensity measurement at a point in time."""
    
    timestamp: datetime = Field(..., description="Measurement timestamp in UTC")
    gco2_per_kwh: float = Field(..., ge=0, description="Carbon intensity in gCO2/kWh")
    
    @field_validator('gco2_per_kwh')
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """Ensure value is finite."""
        if not (v >= 0 and v < float('inf')):
            raise ValueError(f"gco2_per_kwh must be finite and non-negative, got {v}")
        return v


class IntensitySeries(BaseModel):
    """Carbon intensity current reading and forecast series."""
    
    region: str = Field(..., description="Region code (e.g., IN-KA)")
    current: IntensityPoint = Field(..., description="Current intensity reading")
    forecast: List[IntensityPoint] = Field(..., description="Forecast points")


class IntensityProvider(ABC):
    """
    Abstract provider interface for carbon intensity data.
    
    All intensity values are in gCO2/kWh (grams of CO2 per kilowatt-hour).
    All timestamps must be timezone-aware UTC.
    """
    
    @abstractmethod
    async def get_current(self, region: str) -> IntensityPoint:
        """
        Get current carbon intensity for a region.
        
        Args:
            region: Region code (e.g., "IN-KA" for Karnataka, India)
            
        Returns:
            Current intensity point
            
        Raises:
            RuntimeError: If provider is not configured or unavailable
            ValueError: If region is invalid
        """
        pass
    
    @abstractmethod
    async def get_forecast(self, region: str) -> List[IntensityPoint]:
        """
        Get forecasted carbon intensity for a region.
        
        Args:
            region: Region code (e.g., "IN-KA")
            
        Returns:
            List of forecast points (at least 6 points, typically 12 at 5-min intervals)
            
        Raises:
            RuntimeError: If provider is not configured or unavailable
            ValueError: If region is invalid
        """
        pass