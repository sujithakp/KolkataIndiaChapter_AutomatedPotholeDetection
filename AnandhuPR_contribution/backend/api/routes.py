"""
FastAPI routes for pothole detection API.
Handles image and video processing requests.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from services.image_service import process_image_from_url
from services.video_service import process_video_from_url
from tasks.video_tasks import process_video_task
from celery.result import AsyncResult
from billiard.exceptions import WorkerLostError

logger = logging.getLogger(__name__)

app = FastAPI()
router = app.router

class ImageRequest(BaseModel):
    image_url: str

class VideoRequest(BaseModel):
    video_url: str

@router.post("/predict-image/")
async def predict_pothole(request: ImageRequest):
    """
    Process an image to detect potholes.
    """
    try:
        logger.info(f"Processing image: {request.image_url}")
        result = process_image_from_url(request.image_url)
        return result
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/process-video/")
async def process_video(request: VideoRequest):
    """
    Queue a video for pothole detection processing.
    """
    try:
        logger.info(f"Queuing video: {request.video_url}")
        task = process_video_task.delay(request.video_url)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error queuing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error queuing video: {str(e)}")

@router.get("/video-status/{task_id}")
async def get_video_status(task_id: str):
    """
    Retrieve the status of a video processing task.
    """
    try:
        task_result = AsyncResult(task_id)
        if not task_result.backend:
            logger.error("Celery backend is disabled")
            raise HTTPException(status_code=500, detail="Celery backend is disabled")
        
        if task_result.state == 'PENDING':
            return {"task_id": task_id, "status": "PENDING"}
        elif task_result.state == 'STARTED':
            return {"task_id": task_id, "status": "STARTED"}
        elif task_result.state == 'SUCCESS':
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "result": task_result.get()
            }
        elif task_result.state == 'FAILURE':
            try:
                error = str(task_result.get(propagate=False))
            except WorkerLostError:
                error = "Worker crashed unexpectedly"
            logger.error(f"Task {task_id} failed: {error}")
            return {
                "task_id": task_id,
                "status": "FAILURE",
                "error": error
            }
        else:
            return {"task_id": task_id, "status": task_result.state}
    except Exception as e:
        logger.error(f"Error retrieving task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving task status: {str(e)}")