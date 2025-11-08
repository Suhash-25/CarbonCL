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