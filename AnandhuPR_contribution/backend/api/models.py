from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any

class CloudinaryImageRequest(BaseModel):
    image_url: HttpUrl

class CloudinaryVideoRequest(BaseModel):
    video_url: HttpUrl

class PotholeDetection(BaseModel):
    pothole_detected: bool
    severity: float
    image_url: Optional[str] = None 
    detections: List[dict] = []

class VideoAnalysisResult(BaseModel):
    average_severity: float
    damaged_road_percentage: float
    video_url: Optional[str] = None 
    total_potholes_detected: int

class TaskResponse(BaseModel):
    task_id: str
    status: str


class VideoResult(BaseModel):
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None