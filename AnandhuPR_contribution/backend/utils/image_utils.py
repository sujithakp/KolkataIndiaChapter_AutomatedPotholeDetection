import cv2
import base64
import numpy as np
import requests
from fastapi import HTTPException

def download_image_from_url(image_url):
    """Download image from URL and convert to OpenCV format"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Convert to numpy array
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
            
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download or process image: {str(e)}")

def encode_image_to_base64(cv_image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', cv_image)
    return base64.b64encode(buffer).decode('utf-8')