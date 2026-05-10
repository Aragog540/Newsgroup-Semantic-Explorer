"""Configuration management for the Newsgroup Semantic Explorer app."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    app_name: str = Field(default="Newsgroup-Semantic-Explorer", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")

    dataset_type: str = Field(default="20_newsgroups", env="DATASET_TYPE")
    newsgroups_path: str = Field(default="", env="NEWSGROUPS_PATH")

    embedding_type: str = Field(default="lsa", env="EMBEDDING_TYPE")

    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_theta_lsa: float = Field(default=0.60, env="CACHE_THETA_LSA")
    cache_theta_neural: float = Field(default=0.88, env="CACHE_THETA_NEURAL")

    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    allowed_hosts: str = Field(default="*", env="ALLOWED_HOSTS")
    require_https: bool = Field(default=False, env="REQUIRE_HTTPS")

    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")

    log_file_path: str = Field(default="logs/app.log", env="LOG_FILE_PATH")
    log_max_bytes: int = Field(default=10485760, env="LOG_MAX_BYTES")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def validate_at_startup(self):
        """Validate all critical settings at startup."""
        if self.embedding_type not in ["lsa", "neural"]:
            raise ValueError(f"Invalid EMBEDDING_TYPE: {self.embedding_type}. Must be 'lsa' or 'neural'.")

        if self.dataset_type not in ["20_newsgroups", "mini_newsgroups"]:
            raise ValueError(f"Invalid DATASET_TYPE: {self.dataset_type}. Must be '20_newsgroups' or 'mini_newsgroups'.")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid LOG_LEVEL: {self.log_level}")

        if self.cache_theta_lsa <= 0 or self.cache_theta_lsa >= 1:
            raise ValueError("CACHE_THETA_LSA must be between 0 and 1")

        if self.cache_theta_neural <= 0 or self.cache_theta_neural >= 1:
            raise ValueError("CACHE_THETA_NEURAL must be between 0 and 1")

        if self.port < 1 or self.port > 65535:
            raise ValueError("PORT must be between 1 and 65535")

        if self.workers < 1:
            raise ValueError("WORKERS must be at least 1")

        log_dir = Path(self.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get or create settings instance."""
    return Settings()


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure logging with file rotation and proper formatting."""
    from logging.handlers import RotatingFileHandler

    log_dir = Path(settings.log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("newsgroup_search")
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        settings.log_file_path,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
    )
    file_handler.setLevel(getattr(logging, settings.log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
