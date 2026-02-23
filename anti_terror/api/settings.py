"""API server settings via pydantic-settings (reads from env / .env)."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_dsn: str = "postgresql://antiterror:antiterror@localhost:5432/antiterror"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    # Default admin credentials (for first login)
    admin_username: str = "admin"
    admin_password: str = "admin"

    # Pipeline preview streams (MJPEG)
    # Comma-separated list: CAM_01=8081,CAM_02=8082
    preview_streams: str = "CAM_01=8081"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
