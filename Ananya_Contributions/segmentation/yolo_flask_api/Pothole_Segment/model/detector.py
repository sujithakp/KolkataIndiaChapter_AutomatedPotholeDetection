# model/detector.py

# Import necessary libraries
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import yaml
from PIL import Image
from collections import deque
from ultralytics import YOLO


class PotholeDetector:
    _instance = None
    
    @classmethod
    def get_instance(cls, model_path=None):
        """Singleton pattern to ensure model is loaded only once"""
        if cls._instance is None and model_path is not None:
            cls._instance = cls(model_path)
        return cls._instance
    
    def __init__(self, model_path):
        """Initialize the model - should be called only once"""
        print("Loading YOLO model...")
        # Load the best model weights into the YOLO model
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
    
    def detect(self, image):
        """Run inference on the image"""
        predictions = self.model.predict(image)

        # Return the prediction for the first (and only) image
        return predictions
