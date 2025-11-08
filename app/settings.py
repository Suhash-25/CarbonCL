"""Configuration settings for the Compute Metering Service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Metering settings
    MEASURE_POWER_SECS: int = 1
    DATA_DIR: str = "./data"
    LOG_LEVEL: str = "INFO"
    
    # Carbon intensity settings
    EM_BASE_URL: str = "https://api.electricitymap.org/v3"
    EM_API_KEY: str | None = None
    DEFAULT_REGION: str = "IN-KA"
    CACHE_TTL_SECONDS: int = 120
    ENABLE_SIMULATION: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()