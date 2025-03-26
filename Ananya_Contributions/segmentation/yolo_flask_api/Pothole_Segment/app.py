# app.py
from flask import Flask
from api.endpoints import api_bp
from model.detector import PotholeDetector
import config

app = Flask(__name__)

# Initialize the model when the app starts
# This ensures the model is loaded only once when the server starts
PotholeDetector.get_instance(config.MODEL_PATH)

# Register the API blueprint
app.register_blueprint(api_bp, url_prefix='/api/v1')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.PORT, debug=config.DEBUG)
