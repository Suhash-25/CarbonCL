"""Deterministic simulator for carbon intensity data."""

import math
from datetime import datetime, timedelta, timezone
from typing import List

from .intensity_base import IntensityPoint, IntensityProvider


class SimulatorProvider(IntensityProvider):
    """
    Deterministic carbon intensity simulator.
    
    Generates a realistic diurnal pattern:
    - Night (00:00-06:00): Cleaner grid, 250-320 gCO2/kWh
    - Morning (06:00-12:00): Rising intensity
    - Noon-Afternoon (12:00-18:00): Peak solar mixed with demand, 480-560 gCO2/kWh
    - Evening (18:00-24:00): Declining intensity
    
    Uses sinusoidal curve for smooth, repeatable results.
    """
    
    def __init__(self):
        """Initialize the simulator."""
        self.base_intensity = 400.0  # gCO2/kWh baseline
        self.amplitude = 150.0  # Peak variation
        self.min_intensity = 250.0  # Night minimum
    
    def _calculate_intensity(self, dt: datetime) -> float:
        """
        Calculate intensity for a given datetime using sinusoidal pattern.
        
        Args:
            dt: Datetime to calculate intensity for
            
        Returns:
            Carbon intensity in gCO2/kWh
        """
        # Get hour and minute as decimal hours (local time for diurnal pattern)
        hour = dt.hour + dt.minute / 60.0
        
        # Sinusoidal pattern: peak at 15:00 (3 PM), minimum at 03:00 (3 AM)
        # Phase shift: -3 hours (minimum at 3 AM)
        phase_shift = -3.0
        period = 24.0  # 24-hour cycle
        
        # Calculate angle in radians
        angle = 2 * math.pi * (hour + phase_shift) / period
        
        # Calculate intensity: base + amplitude * sin(angle)
        intensity = self.base_intensity + self.amplitude * math.sin(angle)
        
        # Ensure minimum
        intensity = max(intensity, self.min_intensity)
        
        return round(intensity, 2)
    
    async def get_current(self, region: str) -> IntensityPoint:
        """
        Get current simulated carbon intensity.
        
        Args:
            region: Region code (ignored in simulation)
            
        Returns:
            Current intensity point with deterministic value based on time
        """
        now = datetime.now(timezone.utc)
        intensity = self._calculate_intensity(now)
        
        return IntensityPoint(
            timestamp=now,
            gco2_per_kwh=intensity
        )
    
    async def get_forecast(self, region: str) -> List[IntensityPoint]:
        """
        Get forecasted simulated carbon intensity.
        
        Returns 12 points at 5-minute intervals.
        
        Args:
            region: Region code (ignored in simulation)
            
        Returns:
            List of 12 forecast points at 5-minute intervals
        """
        now = datetime.now(timezone.utc)
        forecast = []
        
        for i in range(12):
            future_time = now + timedelta(minutes=5 * i)
            intensity = self._calculate_intensity(future_time)
            
            forecast.append(IntensityPoint(
                timestamp=future_time,
                gco2_per_kwh=intensity
            ))
        
        return forecast