# api/endpoints.py
from flask import Blueprint, request, jsonify, send_file, Response
import io
import base64
from PIL import Image
import cv2
import numpy as np
from model.detector import PotholeDetector
from utils.visualization import draw_detections, calc_damaged_area

api_bp = Blueprint('api', __name__)

@api_bp.route('/detect', methods=['POST'])
def detect_potholes():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read the image from the request
        image_file = request.files['image']
        image = cv2.imread(image_file)
        #image = Image.open(image_file).convert('RGB')
        
        # Get the detector instance (already loaded)
        detector = PotholeDetector.get_instance()
        
        # Run detection
        predictions = detector.detect(image)
        
        
        
        # Filter results based on confidence threshold
        confidence_threshold = float(request.args.get('confidence', 0.5))
        
        
        # Handle different response formats
        response_format = request.args.get('format', 'json_image')
        
        if response_format == 'json':
            # Return just the JSON data
            return jsonify({
                'detections': predictions,
                'image_width': image.shape[1],
                'image_height': image.shape[0]
            })
        
        else:  # 'json_image' or 'image'
            # Draw boxes on the image
            annotated_image = draw_detections(image, predictions)
            damaged_area = calc_damaged_area(predictions)
            
            # Convert annotated image to base64
            _, im_arr = cv2.imencode('.jpg', annotated_image)  # im_arr: image in Numpy one-dim array format.
            im_bytes = im_arr.tobytes()
            im_b64 = base64.b64encode(im_bytes)

            if response_format == 'image':
                # Return just the image
                return send_file(annotated_image, mimetype='image/png')
            else:  # 'json_image'
                # Return both JSON and image
                # First save the image to a file
                from flask import current_app
                import uuid
                import os
                
                filename = f"{uuid.uuid4()}.png"
                file_path = os.path.join(current_app.root_path, 'uploads', filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                #annotated_image.save(file_path)
                cv2.imwrite(file_path, annotated_image)
                
                return jsonify({
                    #'detections': predictions,
                    'image_width': image.shape[1],
                    'image_height': image.shape[0],
                    'annotated_image_url': f"/uploads/{filename},",
                    'annotated_image_base64': f"data:image/png;base64,{im_b64}",
                    'damaged_area': damaged_area
                })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
