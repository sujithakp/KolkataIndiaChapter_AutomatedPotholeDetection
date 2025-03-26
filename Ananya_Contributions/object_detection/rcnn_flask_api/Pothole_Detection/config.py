# config.py
import os

# Model settings
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/weights/pothole_rcnn_best.pth')


LOW_THRESHOLD = os.environ.get('LOW_THRESHOLD', 10000)
MEDIUM_THRESHOLD = os.environ.get('LOW_THRESHOLD', 40000)

# Server settings
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# API settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
