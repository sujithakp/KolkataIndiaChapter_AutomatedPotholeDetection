# utils/visualization.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from bbox_utils.bbox_2d import BoundingBox
import config

def draw_detections(image, detections, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the image
    
    Args:
        image: PIL Image object
        detections: List of detection dictionaries with 'box', 'score', and 'label'
        color: RGB tuple for the bounding box color
        thickness: Line thickness for the bounding box
    
    Returns:
        PIL Image with bounding boxes drawn
    """
    # Create a copy of the image
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each detection
    for det in detections:
        box = det['box']
        score = det['score']
        label = det['label']
        
        # Convert box coordinates to integers
        box = [int(b) for b in box]

        #calculate severity
        x1, y1, x2, y2 = box
        bbox1 = BoundingBox.from_xyxy(np.array([x1,y1]), np.array([x2,y2]))
        area = bbox1.height * bbox1.width

        sev_level = 'unknown'
        if area <= config.LOW_THRESHOLD:
            sev_level = 'low'
        elif area <= config.MEDIUM_THRESHOLD:
            sev_level = 'medium'
        elif area > config.MEDIUM_THRESHOLD:
            sev_level = 'high'

        
        
        # Draw the bounding box
        draw.rectangle(box, outline=color, width=thickness)
        
        # Draw label and score
        #text = f"Pothole: {score:.2f}"
        text = f"Severity: {sev_level}, Area: {area:.2f}"
        text_position = (box[0], box[1] - 20)
        
        # Add a background to the text for better visibility
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        draw.rectangle(
            [text_position[0], text_position[1], 
             text_position[0] + text_width, text_position[1] + text_height],
            fill=(0, 0, 0)
        )
        
        # Draw the text
        draw.text(text_position, text, font=font, fill=(255, 255, 255))
    
    return result_image
