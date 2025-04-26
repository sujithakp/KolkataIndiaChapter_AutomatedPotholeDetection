"""
Utilities for initializing the YOLO model with automatic device selection.
Supports NVIDIA CUDA, Apple MPS, and CPU fallback.
"""

import torch
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

def get_device():
    """
    Determine the best available device for YOLO inference.

    Returns:
        str: Device name ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using NVIDIA CUDA GPU (device: {torch.cuda.get_device_name(0)})")
    else:
        device = 'cpu'
        logger.info("No GPU detected, falling back to CPU")
    return device

def get_model(model_path="yolov11_weight/best_remya.pt"):  # Update path
    """
    Initialize the YOLO model and assign it to the best available device.

    Args:
        model_path (str): Path to the YOLO model weights.

    Returns:
        YOLO: Initialized YOLO model on the selected device.
    """
    try:
        device = get_device()
        model = YOLO(model_path)
        model.to(device)
        logger.info(f"YOLO model loaded on device: {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        logger.info("Falling back to CPU")
        model = YOLO(model_path)
        model.to('cpu')
        logger.info("YOLO model loaded on device: cpu")
        return model