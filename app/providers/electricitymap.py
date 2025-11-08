"""ElectricityMap API client for real carbon intensity data."""

import logging
from datetime import datetime, timezone
from typing import List

import httpx

from .intensity_base import IntensityPoint, IntensityProvider

logger = logging.getLogger(__name__)


class ElectricityMapProvider(IntensityProvider):
    """
    ElectricityMap API provider for carbon intensity data.
    
    Uses the ElectricityMap v3 API to fetch real-time and forecasted
    carbon intensity data for electricity grids worldwide.
    """
    
    def __init__(self, base_url: str, api_key: str):
        """
        Initialize the ElectricityMap provider.
        
        Args:
            base_url: Base URL for ElectricityMap API
            api_key: API key for authentication
            
        Raises:
            RuntimeError: If API key is not provided
        """
        if not api_key:
            raise RuntimeError(
                "ElectricityMap API key is required. "
                "Set EM_API_KEY environment variable or enable simulation mode."
            )
        
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            timeout=10.0,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
    
    async def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Make an authenticated request to ElectricityMap API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            RuntimeError: If request fails
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = await self.client.get(url, params=params)
            
            if response.status_code != 200:
                error_msg = f"ElectricityMap API error: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error']}"
                except Exception:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            return response.json()
        
        except httpx.TimeoutException:
            raise RuntimeError("ElectricityMap API timeout after 10 seconds")
        except httpx.RequestError as e:
            raise RuntimeError(f"ElectricityMap API request failed: {str(e)}")
    
    def _parse_intensity(self, data: dict) -> float:
        """
        Parse carbon intensity from API response.
        
        Args:
            data: API response data
            
        Returns:
            Carbon intensity in gCO2/kWh
            
        Raises:
            RuntimeError: If intensity data is missing or invalid
        """
        # ElectricityMap may return intensity in different fields
        intensity = data.get('carbonIntensity') or data.get('carbon_intensity')
        
        if intensity is None:
            raise RuntimeError(
                f"No carbon intensity data available for this region. "
                f"Response: {str(data)[:200]}"
            )
        
        try:
            intensity_float = float(intensity)
            if intensity_float < 0 or not (intensity_float < float('inf')):
                raise ValueError(f"Invalid intensity value: {intensity_float}")
            return intensity_float
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Invalid carbon intensity format: {intensity} - {e}")
    
    def _parse_timestamp(self, data: dict) -> datetime:
        """
        Parse timestamp from API response.
        
        Args:
            data: API response data
            
        Returns:
            Timezone-aware UTC datetime
        """
        timestamp_str = data.get('datetime') or data.get('timestamp')
        
        if not timestamp_str:
            # Fallback to current time if no timestamp provided
            return datetime.now(timezone.utc)
        
        try:
            # Parse ISO format timestamp
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Ensure UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            
            return dt
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp_str}: {e}")
            return datetime.now(timezone.utc)
    
    async def get_current(self, region: str) -> IntensityPoint:
        """
        Get current carbon intensity from ElectricityMap.
        
        Args:
            region: Region/zone code (e.g., "IN-KA")
            
        Returns:
            Current intensity point
            
        Raises:
            RuntimeError: If API request fails
            ValueError: If region is invalid
        """
        if not region or not region.strip():
            raise ValueError("Region code cannot be empty")
        
        region = region.strip()
        
        # ElectricityMap v3 endpoint for carbon intensity
        endpoint = "carbon-intensity/latest"
        params = {"zone": region}
        
        try:
            data = await self._make_request(endpoint, params)
            
            intensity = self._parse_intensity(data)
            timestamp = self._parse_timestamp(data)
            
            logger.info(
                f"ElectricityMap current for {region}: {intensity} gCO2/kWh at {timestamp}"
            )
            
            return IntensityPoint(
                timestamp=timestamp,
                gco2_per_kwh=intensity
            )
        
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to get current intensity: {str(e)}")
    
    async def get_forecast(self, region: str) -> List[IntensityPoint]:
        """
        Get forecasted carbon intensity from ElectricityMap.
        
        Args:
            region: Region/zone code (e.g., "IN-KA")
            
        Returns:
            List of forecast points (typically 24-48 hours ahead)
            
        Raises:
            RuntimeError: If API request fails
            ValueError: If region is invalid
        """
        if not region or not region.strip():
            raise ValueError("Region code cannot be empty")
        
        region = region.strip()
        
        # ElectricityMap v3 endpoint for forecast
        endpoint = "carbon-intensity/forecast"
        params = {"zone": region}
        
        try:
            data = await self._make_request(endpoint, params)
            
            # Parse forecast array
            forecast_data = data.get('forecast', [])
            
            if not forecast_data:
                # Fallback: if no forecast, return current value repeated
                logger.warning(f"No forecast data for {region}, using current value")
                current = await self.get_current(region)
                return [current] * 12
            
            forecast = []
            for point in forecast_data:
                try:
                    intensity = self._parse_intensity(point)
                    timestamp = self._parse_timestamp(point)
                    
                    forecast.append(IntensityPoint(
                        timestamp=timestamp,
                        gco2_per_kwh=intensity
                    ))
                except Exception as e:
                    logger.warning(f"Skipping invalid forecast point: {e}")
                    continue
            
            # Ensure we have at least 6 points
            if len(forecast) < 6:
                logger.warning(
                    f"Only {len(forecast)} forecast points for {region}, "
                    f"expected at least 6"
                )
                if not forecast:
                    # Last resort: use current value
                    current = await self.get_current(region)
                    return [current] * 12
            
            # Sort by timestamp
            forecast.sort(key=lambda p: p.timestamp)
            
            logger.info(
                f"ElectricityMap forecast for {region}: {len(forecast)} points"
            )
            
            return forecast
        
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to get forecast: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()