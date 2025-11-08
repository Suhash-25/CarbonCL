"""Carbon intensity API router with caching."""

import logging
from typing import List, Optional

from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from .providers.electricitymap import ElectricityMapProvider
from .providers.intensity_base import IntensityPoint, IntensityProvider
from .providers.simulator import SimulatorProvider
from .settings import settings

logger = logging.getLogger(__name__)

# Cache for intensity data
# Key format: (region, endpoint_type) -> cached response
intensity_cache = TTLCache(maxsize=100, ttl=settings.CACHE_TTL_SECONDS)

# Provider instance (singleton)
_provider_instance: Optional[IntensityProvider] = None


class CurrentIntensityResponse(BaseModel):
    """Response for current carbon intensity."""
    
    region: str = Field(..., description="Region code")
    gco2_per_kwh: float = Field(..., description="Carbon intensity in gCO2/kWh")
    timestamp: str = Field(..., description="Timestamp in ISO 8601 format")
    source: str = Field(..., description="Data source: 'electricitymap' or 'simulator'")


class ForecastPointResponse(BaseModel):
    """A single forecast point."""
    
    timestamp: str = Field(..., description="Timestamp in ISO 8601 format")
    gco2_per_kwh: float = Field(..., description="Carbon intensity in gCO2/kWh")


class ForecastIntensityResponse(BaseModel):
    """Response for forecasted carbon intensity."""
    
    region: str = Field(..., description="Region code")
    points: List[ForecastPointResponse] = Field(..., description="Forecast data points")
    source: str = Field(..., description="Data source: 'electricitymap' or 'simulator'")


def get_provider() -> IntensityProvider:
    """
    Get the configured intensity provider.
    
    Returns simulator if:
    - ENABLE_SIMULATION is True, OR
    - No ElectricityMap API key is configured
    
    Returns:
        Intensity provider instance
    """
    global _provider_instance
    
    if _provider_instance is None:
        if settings.ENABLE_SIMULATION:
            logger.info("Using SimulatorProvider (ENABLE_SIMULATION=True)")
            _provider_instance = SimulatorProvider()
        elif not settings.EM_API_KEY:
            logger.warning(
                "No EM_API_KEY configured, falling back to SimulatorProvider"
            )
            _provider_instance = SimulatorProvider()
        else:
            logger.info("Using ElectricityMapProvider with real API")
            _provider_instance = ElectricityMapProvider(
                base_url=settings.EM_BASE_URL,
                api_key=settings.EM_API_KEY
            )
    
    return _provider_instance


router = APIRouter(
    prefix="/intensity",
    tags=["carbon-intensity"]
)


@router.get(
    "/current",
    response_model=CurrentIntensityResponse,
    summary="Get current carbon intensity",
    description="Get current carbon intensity for a region in gCO2/kWh"
)
async def get_current_intensity(
    region: Optional[str] = Query(
        None,
        description=f"Region code (e.g., 'IN-KA'). Defaults to {settings.DEFAULT_REGION}"
    ),
    provider: IntensityProvider = Depends(get_provider)
) -> CurrentIntensityResponse:
    """
    Get current carbon intensity for a region.
    
    Args:
        region: Region code (optional, defaults to configured DEFAULT_REGION)
        provider: Intensity provider (dependency injected)
        
    Returns:
        Current intensity data with source information
        
    Raises:
        HTTPException: 400 for invalid region, 502 for provider errors
    """
    # Use default region if not specified
    if not region:
        region = settings.DEFAULT_REGION
    
    # Validate region
    region = region.strip()
    if not region:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Region code cannot be empty"
        )
    
    # Check cache
    cache_key = (region, "current")
    if cache_key in intensity_cache:
        logger.info(f"Cache hit for current intensity: {region}")
        return intensity_cache[cache_key]
    
    # Fetch from provider
    try:
        logger.info(f"Fetching current intensity for {region}")
        intensity_point = await provider.get_current(region)
        
        # Determine source
        source = "simulator" if isinstance(provider, SimulatorProvider) else "electricitymap"
        
        response = CurrentIntensityResponse(
            region=region,
            gco2_per_kwh=intensity_point.gco2_per_kwh,
            timestamp=intensity_point.timestamp.isoformat(),
            source=source
        )
        
        # Cache the response
        intensity_cache[cache_key] = response
        logger.info(f"Cached current intensity for {region}")
        
        return response
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Provider error: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error getting current intensity for {region}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.get(
    "/forecast",
    response_model=ForecastIntensityResponse,
    summary="Get forecasted carbon intensity",
    description="Get forecasted carbon intensity for a region (typically 12 points at 5-min intervals)"
)
async def get_forecast_intensity(
    region: Optional[str] = Query(
        None,
        description=f"Region code (e.g., 'IN-KA'). Defaults to {settings.DEFAULT_REGION}"
    ),
    provider: IntensityProvider = Depends(get_provider)
) -> ForecastIntensityResponse:
    """
    Get forecasted carbon intensity for a region.
    
    Args:
        region: Region code (optional, defaults to configured DEFAULT_REGION)
        provider: Intensity provider (dependency injected)
        
    Returns:
        Forecast data with multiple time points and source information
        
    Raises:
        HTTPException: 400 for invalid region, 502 for provider errors
    """
    # Use default region if not specified
    if not region:
        region = settings.DEFAULT_REGION
    
    # Validate region
    region = region.strip()
    if not region:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Region code cannot be empty"
        )
    
    # Check cache
    cache_key = (region, "forecast")
    if cache_key in intensity_cache:
        logger.info(f"Cache hit for forecast: {region}")
        return intensity_cache[cache_key]
    
    # Fetch from provider
    try:
        logger.info(f"Fetching forecast for {region}")
        forecast_points = await provider.get_forecast(region)
        
        # Determine source
        source = "simulator" if isinstance(provider, SimulatorProvider) else "electricitymap"
        
        # Convert to response format
        points = [
            ForecastPointResponse(
                timestamp=point.timestamp.isoformat(),
                gco2_per_kwh=point.gco2_per_kwh
            )
            for point in forecast_points
        ]
        
        response = ForecastIntensityResponse(
            region=region,
            points=points,
            source=source
        )
        
        # Cache the response
        intensity_cache[cache_key] = response
        logger.info(f"Cached forecast for {region} ({len(points)} points)")
        
        return response
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Provider error: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error getting forecast for {region}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )