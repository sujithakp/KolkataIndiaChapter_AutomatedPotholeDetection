"""
Centralized configuration for Cloudinary to ensure consistent setup across Django, FastAPI, and Celery.
Loads environment variables from .env and initializes Cloudinary credentials.
"""

import os
import cloudinary
from dotenv import load_dotenv
import logging

# Configure logging
logger = logging.getLogger(__name__)

def init_cloudinary():
    """
    Initialize Cloudinary configuration using environment variables.
    
    Raises:
        ValueError: If any required Cloudinary credential is missing.
    """
    load_dotenv()  # Load .env file from project root
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")

    # Validate credentials
    if not all([cloud_name, api_key, api_secret]):
        missing = [var for var in ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"] if not os.getenv(var)]
        logger.error(f"Missing Cloudinary config: {', '.join(missing)}")
        raise ValueError(f"Missing Cloudinary config: {', '.join(missing)}")

    # Configure Cloudinary
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True
    )
    logger.info(f"Cloudinary configured with cloud_name: {cloud_name}")

# Initialize on module import to ensure config is set for all processes
init_cloudinary()