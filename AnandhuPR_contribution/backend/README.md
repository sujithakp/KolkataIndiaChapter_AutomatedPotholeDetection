# Pothole Detection API

This API detects potholes in images and videos using a YOLOv11 model. It accepts Cloudinary URLs for both images and videos and returns detection results with visualizations.

## API Endpoints

### 1. Image Processing
**Endpoint:** `/predict-image/`
**Method:** POST
**Request Body:**
```json
{
  "image_url": "https://your-cloudinary-url.com/image.jpg"
}
```

### 2. Video Processing
**Endpoint:** `/process-video/`
**Method:** POST
**Request Body:**
```json
{
  "video_url": "https://your-cloudinary-url.com/video.mp4"
}
```

## Example Usage

```python
import requests
import json

# Process an image
response = requests.post(
    "https://your-huggingface-space.hf.space/predict-image/",
    json={"image_url": "https://your-cloudinary-url.com/image.jpg"}
)
result = response.json()

# Process a video
response = requests.post(
    "https://your-huggingface-space.hf.space/process-video/",
    json={"video_url": "https://your-cloudinary-url.com/video.mp4"}
)
result = response.json()
```

## Model Details

This API uses a YOLOv11 model trained to detect and segment potholes in road images and videos.