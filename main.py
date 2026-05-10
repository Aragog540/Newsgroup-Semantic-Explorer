"""ASGI entry point for the Newsgroup Semantic Explorer application."""

from app import app, logger
from config import get_settings

settings = get_settings()

if __name__ == "__main__":
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Listening on {settings.host}:{settings.port}")
    logger.info(f"Workers: {settings.workers}")

