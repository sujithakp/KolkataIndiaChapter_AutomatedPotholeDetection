"""
Celery tasks for asynchronous video processing.
"""

from celery import Celery
from services.video_service import process_video_from_url
import logging

logger = logging.getLogger(__name__)

# app = Celery(
#     'tasks',
#     broker='redis://localhost:6379/0',
#     backend='redis://localhost:6379/0'
# )

# for redis in docker

app = Celery(
    'pothole_detection',
    broker='redis://redis:6379/0',  # Use the service name 'redis' instead of localhost
    backend='redis://redis:6379/0'  # Same for backend
)



app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,
    task_track_started=True,
)

@app.task(bind=True)
def process_video_task(self, video_url):
    """
    Process a video to detect potholes and return results.

    Args:
        video_url (str): URL of the video to process.

    Returns:
        dict: Processing results.
    """
    logger.info(f"Received task: {self.request.id} for video: {video_url}")
    try:
        logger.info("Starting video processing...")
        result = process_video_from_url(video_url)
        logger.info(f"Task completed: {self.request.id}, result: {result}")
        return result
    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {str(e)}")
        raise self.retry(exc=e, countdown=10, max_retries=3)