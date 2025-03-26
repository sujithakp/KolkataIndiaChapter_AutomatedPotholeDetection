# config.py
import os

# Model settings
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/weights/yolo11m_best_dev6.pt')



# Server settings
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# API settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
