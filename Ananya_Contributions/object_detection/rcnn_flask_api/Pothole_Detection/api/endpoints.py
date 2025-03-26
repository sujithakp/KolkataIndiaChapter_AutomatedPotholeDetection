# api/endpoints.py
from flask import Blueprint, request, jsonify, send_file, Response
import io
import base64
from PIL import Image
import numpy as np
from model.detector import PotholeDetector
from utils.visualization import draw_detections

api_bp = Blueprint('api', __name__)

@api_bp.route('/detect', methods=['POST'])
def detect_potholes():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read the image from the request
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')
        
        # Get the detector instance (already loaded)
        detector = PotholeDetector.get_instance()
        
        # Run detection
        predictions = detector.detect(image)
        
        # Process the prediction results
        boxes = predictions['boxes'].cpu().numpy().tolist()
        scores = predictions['scores'].cpu().numpy().tolist()
        labels = predictions['labels'].cpu().numpy().tolist()
        
        # Filter results based on confidence threshold
        confidence_threshold = float(request.args.get('confidence', 0.5))
        filtered_results = []
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                filtered_results.append({
                    'box': box,  # [x1, y1, x2, y2]
                    'score': score,
                    'label': label
                })
        
        # Handle different response formats
        response_format = request.args.get('format', 'json_image')
        
        if response_format == 'json':
            # Return just the JSON data
            return jsonify({
                'detections': filtered_results,
                'image_width': image.width,
                'image_height': image.height
            })
        
        else:  # 'json_image' or 'image'
            # Draw boxes on the image
            annotated_image = draw_detections(image, filtered_results)
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            annotated_image.save(img_byte_arr, format='PNG')
            img_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            img_byte_arr.seek(0)
            
            if response_format == 'image':
                # Return just the image
                return send_file(img_byte_arr, mimetype='image/png')
            else:  # 'json_image'
                # Return both JSON and image
                # First save the image to a file
                from flask import current_app
                import uuid
                import os
                
                filename = f"{uuid.uuid4()}.png"
                file_path = os.path.join(current_app.root_path, 'static', filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                annotated_image.save(file_path)
                
                return jsonify({
                    'detections': filtered_results,
                    'image_width': image.width,
                    'image_height': image.height,
                    'annotated_image_url': f"/static/{filename},",
                    'annotated_image_base64': f"data:image/png;base64,{img_str}"
                })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
