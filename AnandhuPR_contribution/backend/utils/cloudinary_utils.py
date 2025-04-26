"""
Utility functions for uploading images and videos to Cloudinary with retry mechanism.
"""

import cloudinary
import cloudinary.uploader
import logging
from retrying import retry
import requests

logger = logging.getLogger(__name__)

def is_transient_error(exception):
    """
    Determine if an exception is transient and worth retrying.
    """
    if isinstance(exception, (requests.exceptions.RequestException, cloudinary.exceptions.Error)):
        return True
    return False

@retry(
    stop_max_attempt_number=3,
    wait_fixed=2000,
    retry_on_exception=is_transient_error
)
def upload_image_to_cloudinary(image_path, public_id=None):
    """
    Upload an image to Cloudinary with specified parameters.

    Args:
        image_path (str): Path to the image file.
        public_id (str, optional): Custom public ID for the image.

    Returns:
        str: URL of the uploaded image.

    Raises:
        Exception: If upload fails after retries.
    """
    try:
        logger.info(f"Uploading image: {image_path}")
        upload_params = {
            "resource_type": "image",
            "folder": "pothole_detection",
            "overwrite": True,
            "invalidate": True,
        }
        if public_id:
            upload_params["public_id"] = public_id
        
        upload_result = cloudinary.uploader.upload(
            image_path,
            **upload_params
        )
        image_url = upload_result.get("secure_url")
        if not image_url:
            raise Exception("Upload succeeded but no URL returned")
        logger.info(f"Uploaded image to: {image_url}")
        return image_url
    
    except Exception as e:
        logger.error(f"Image upload failed: {str(e)}")
        raise Exception(f"Failed to upload image to Cloudinary: {str(e)}")

@retry(
    stop_max_attempt_number=3,
    wait_fixed=2000,
    retry_on_exception=is_transient_error
)
def upload_video_to_cloudinary(video_path, public_id=None):
    """
    Upload a video to Cloudinary with specified parameters.

    Args:
        video_path (str): Path to the video file.
        public_id (str, optional): Custom public ID for the video.

    Returns:
        str: URL of the uploaded video.

    Raises:
        Exception: If upload fails after retries.
    """
    try:
        logger.info(f"Uploading video: {video_path}")
        upload_params = {
            "resource_type": "video",
            "folder": "pothole_detection",
            "overwrite": True,
            "invalidate": True,
            "format": "mp4",
            "video_codec": "h264",
        }
        if public_id:
            upload_params["public_id"] = public_id
        
        upload_result = cloudinary.uploader.upload(
            video_path,
            **upload_params
        )
        video_url = upload_result.get("secure_url")
        if not video_url:
            raise Exception("Upload succeeded but no URL returned")
        logger.info(f"Uploaded video to: {video_url}")
        return video_url
    
    except Exception as e:
        logger.error(f"Video upload failed: {str(e)}")
        raise Exception(f"Failed to upload video to Cloudinary: {str(e)}")