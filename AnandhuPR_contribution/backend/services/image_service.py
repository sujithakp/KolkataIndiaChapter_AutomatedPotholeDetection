"""
Service module for processing images to detect potholes using a YOLO model.
This module downloads images from URLs, performs inference, uploads results to Cloudinary,
and sends annotated reports via Telegram.
"""

import os
import cv2
import uuid
import tempfile
import numpy as np
import requests
import logging
import asyncio
from utils.model_utils import get_model
from utils.cloudinary_utils import upload_image_to_cloudinary
from services.telegram_bot import send_telegram_message_with_photo


# Configure logging for the service module

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('image_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Global model instance to avoid reloading

_MODEL = None

def get_cached_model():
    """
    Load and cache the YOLO model for repeated use.
    Returns:
        model (object): Loaded YOLO model instance.
    """
    global _MODEL
    if _MODEL is None:
        logger.info("Loading YOLO model")
        _MODEL = get_model()
    return _MODEL

def generate_synthetic_coordinates():
    """
    Generate random latitude and longitude coordinates.
    Simulates GPS coordinates for pothole detections.
    Returns:
        tuple: (latitude, longitude) as floats
    """
    logger.debug("Generating synthetic coordinates")
    lat = round(np.random.uniform(-90, 90), 6)
    lon = round(np.random.uniform(-180, 180), 6)
    return lat, lon

def model_predict(model, image):
    """
    Run the YOLO model inference on the given image.
    Args:
        model (object): YOLO model instance.
        image (np.ndarray): Input image array.
    Returns:
        results (list): List of model prediction results.
    """
    logger.debug("Running model prediction")
    if model is None:
        logger.error("Model is not loaded properly")
        raise ValueError("Model is not loaded properly.")
    try:
        return model.predict(image, show=False, conf=0.3, imgsz=640)
    except Exception as e:
        logger.error(f"Model prediction failed: {str(e)}", exc_info=True)
        raise

async def process_image_from_url_async(image_url):
    """
    Asynchronously download an image, perform pothole detection,
    annotate the image, upload the result to Cloudinary,
    and send a notification via Telegram.
    
    Args:
        image_url (str): URL of the image to be processed.
    
    Returns:
        dict: Inference results including severity, Cloudinary URL,
              potholes detected, location, and Telegram status.
    """
    temp_image_path = None
    processed_image_path = None
    
    try:
        # Download the image from URL
        logger.info(f"Downloading image from URL: {image_url}")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            temp_image.write(response.content)
            temp_image_path = temp_image.name
        
        logger.info(f"Saved temporary image to: {temp_image_path}")
        
        # Read the image into memory
        image = cv2.imread(temp_image_path)
        if image is None:
            logger.error("Unable to read image file")
            raise ValueError("Unable to read image file")
        
        # Generate synthetic location coordinates
        lat, lon = generate_synthetic_coordinates()
        logger.debug(f"Location: {lat}, {lon}")
        
        # Resize image if larger than expected (for speed)
        max_size = 640
        height, width = image.shape[:2]
        scale = min(max_size / width, max_size / height)
        if scale < 1:
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        logger.debug(f"Image size after resizing: {image.shape[1]}x{image.shape[0]}")
        
        # Load (cached) YOLO model
        model = get_cached_model()
        logger.info("Using YOLO model")
        
        # Perform model inference
        logger.info("Running pothole detection")
        results_list = model_predict(model, image)
        
        # If no potholes detected, prepare empty mask
        if not results_list:
            logger.info("No potholes detected")
            pothole_percentage = 0
            detected_potholes = []
            binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            # Process the detection results
            results = results_list[0]
            binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            detected_potholes = []
            
            # Extract bounding boxes
            if hasattr(results, 'boxes') and results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    detected_potholes.append((x1, y1, x2, y2, confidence))
                    logger.debug(f"Pothole detected at ({x1}, {y1}, {x2}, {y2}) with confidence {confidence:.2f}")
            
            # Extract segmentation masks if available
            if hasattr(results, 'masks') and results.masks is not None and hasattr(results.masks, 'data'):
                mask_data = results.masks.data.cpu().numpy()
                for mask in mask_data:
                    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    binary_mask = cv2.bitwise_or(binary_mask, mask_resized)
            
            logger.debug(f"Mask sum (non-zero pixels): {np.sum(binary_mask)}")
            
            # Estimate the pothole severity percentage
            road_pixels = np.sum((image[:, :, 0] > 50) | (image[:, :, 1] > 50) | (image[:, :, 2] > 50))
            pothole_pixels = np.sum(binary_mask > 0)
            pothole_percentage = (pothole_pixels / road_pixels * 100) if road_pixels > 0 else 0
        
        processed_image = image.copy()
        
        # Annotate processed image with detection results
        # cv2.putText(processed_image, f"Severity: {pothole_percentage:.2f}%", (20, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(processed_image, f"Pothole Area: {pothole_percentage:.2f}%", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(processed_image, f"Potholes: {len(detected_potholes)}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw pothole contours on the image
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(processed_image, contours, -1, (0, 255, 0), 2)
        
        #  Draw bounding boxes (commented out )
        # for x1, y1, x2, y2, confidence in detected_potholes:
        #     cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(processed_image, f"{confidence:.2f}", (x1, y1 - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Apply a magenta overlay to detected pothole areas
        mask_colored = np.zeros_like(processed_image)
        mask_colored[binary_mask > 0] = (255, 0, 128)  # Magenta color
        processed_image = cv2.addWeighted(processed_image, 0.9, mask_colored, 0.1, 0)
        
        # Save the annotated image to a temporary file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_proc_file:
                processed_image_path = temp_proc_file.name
            cv2.imwrite(processed_image_path, processed_image)
            logger.info(f"Saved processed image to: {processed_image_path}")
            logger.info(f"Processed image file exists: {os.path.exists(processed_image_path)}")
        except Exception as e:
            logger.error(f"Failed to save processed image: {str(e)}", exc_info=True)
            raise
        
        # Upload the processed image to Cloudinary
        logger.info("Uploading processed image to Cloudinary")
        image_url = upload_image_to_cloudinary(
            processed_image_path,
            public_id=f"pothole_detection/image_{uuid.uuid4()}"
        )
        logger.info(f"Uploaded image available at: {image_url}")
        
        # Prepare a detailed Telegram report message
        report_message = (
            f"**Pothole Detection Report**\n"
            f"• Potholes detected: {len(detected_potholes)}\n"
            f"• Severity: {pothole_percentage:.2f}%\n"
            f"• Location: {lat}, {lon}\n"
            f"• Processed Image URL: {image_url}\n"
        )
        
        # Send Telegram message with the processed image
        logger.info("Sending Telegram notification")
        if processed_image_path and os.path.exists(processed_image_path) and os.path.getsize(processed_image_path) > 0:
            try:
                telegram_success = await send_telegram_message_with_photo(report_message, photo_path=processed_image_path)
                logger.info(f"Telegram message sent successfully: {telegram_success}")
                telegram_status = "Success" if telegram_success else "Failed: Unknown error"
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {str(e)}", exc_info=True)
                telegram_status = f"Failed: {str(e)}"
        else:
            logger.error("Processed image not available for sending")
            try:
                telegram_success = await send_telegram_message_with_photo(report_message)
                telegram_status = "Success (text only)" if telegram_success else "Failed: Image not valid"
            except Exception as e:
                logger.error(f"Failed to send text-only Telegram message: {str(e)}", exc_info=True)
                telegram_status = f"Failed: {str(e)}"
        
        # Prepare the final result dictionary
        
        
        result = {
            "severity": float(round(pothole_percentage, 2)),
            "pothole_percentage": float(round(pothole_percentage, 2)),
            "image_url": image_url,
            "potholes_detected": len(detected_potholes),
            "location": {"lat": lat, "lon": lon},
            "telegram_status": telegram_status
        }
        logger.info(f"Final result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Always clean up temporary files
        for path in [temp_image_path, processed_image_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"Deleted temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {path}: {str(e)}")

def process_image_from_url(image_url):
    """
    Synchronous wrapper around the asynchronous image processing function.
    
    Args:
        image_url (str): URL of the image to be processed.
    
    Returns:
        dict: Processed results dictionary.
    """
    return asyncio.run(process_image_from_url_async(image_url))
