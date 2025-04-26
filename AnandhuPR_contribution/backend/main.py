"""
FastAPI application entry point for the Pothole Detection API.
Sets up CORS, initializes the YOLO model, configures Cloudinary, and includes API routes.
Designed for deployment on Hugging Face Spaces.
"""

import uvicorn
import nest_asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from api.routes import router
from utils.model_utils import get_model
from config import init_cloudinary  # Initialize Cloudinary configuration

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Apply nest_asyncio to support asyncio in Jupyter-like environments (e.g., Hugging Face Spaces)
nest_asyncio.apply()

# Create FastAPI app with a descriptive title
app = FastAPI(title="Pothole Detection API")

# Enable CORS to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Configure services at startup
@app.on_event("startup")
async def startup_event():
    """
    Initialize services when the FastAPI app starts.
    Loads the YOLO model and ensures Cloudinary is configured.
    """
    logger.info("Starting FastAPI application setup")
    get_model()  # Load YOLO model for pothole detection
    init_cloudinary()   # Configure Cloudinary for image and video uploads
    logger.info("Startup completed: Model and Cloudinary initialized")

# Include API routes from the routes module
app.include_router(router)

logger.info("FastAPI application started")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000, 
        reload=True  # Enable auto-reload for development
    )