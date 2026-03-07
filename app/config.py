from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://localhost:5432/fittd"
    s3_bucket: str = "fittd-models"
    aws_region: str = "us-east-1"
    nike_base_url: str = "https://www.nike.com"
    cors_origins: list[str] = ["*"]
    smplx_model_path: str = "./models/smplx"

    model_config = {"env_file": ".env", "env_prefix": "FITTD_"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
