"""
Service module for processing videos to detect potholes using a YOLO model.
Downloads videos from URLs, applies frame-skipping for efficiency, and uploads results to Cloudinary.
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
from utils.cloudinary_utils import upload_video_to_cloudinary
from services.telegram_bot import send_telegram_message_with_photo

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_service.log')
    ]
)
logger = logging.getLogger(__name__)

def generate_synthetic_coordinates():
    """
    Generate random latitude and longitude coordinates.
    """
    logger.debug("Generating synthetic coordinates")
    lat = round(np.random.uniform(-90, 90), 6)
    lon = round(np.random.uniform(-180, 180), 6)
    return lat, lon

def model_predict(model, frame):
    """
    Helper function to run model prediction on a frame.
    """
    logger.debug("Running model prediction")
    if model is None:
        logger.error("Model is not loaded properly")
        raise ValueError("Model is not loaded properly.")
    try:
        return model.predict(frame, show=False, conf=0.3, imgsz=640)
    except Exception as e:
        logger.error(f"Model prediction failed: {str(e)}", exc_info=True)
        raise

def validate_video_file(video_path):
    """
    Validate that the video file is readable and properly formatted.
    """
    logger.debug(f"Validating video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        logger.error("Cannot open video file")
        raise ValueError("Cannot open video file")
    ret, _ = cap.read()
    cap.release()
    if not ret:
        logger.error("Video file is empty or corrupted")
        raise ValueError("Video file is empty or corrupted")
    logger.debug("Video file validated successfully")
    return True

def process_video_from_url(video_url):
    """
    Download a video from a URL and process it to detect potholes.
    """
    temp_video_path = None
    logger.info(f"Downloading video from URL: {video_url}")
    
    try:
        response = requests.get(video_url, timeout=10)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(response.content)
            temp_video_path = temp_video.name
        
        logger.info(f"Saved temporary video to: {temp_video_path}")
        
        model = get_model()
        logger.info("Loaded YOLO model successfully")
        
        return process_video_with_model(temp_video_path, model, skip_frames=2) #frame  skipping
        
    except requests.RequestException as e:
        logger.error(f"Network error while downloading video: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to process video: {str(e)}", exc_info=True)
        raise
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.debug(f"Deleted temporary file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")

def process_video_with_model(video_path, model, skip_frames=3):
    """
    Process a video file with a YOLO model to detect potholes.
    """
    processed_video_path = None
    cap = None
    out = None
    temp_frame_path = None
    
    try:
        logger.info(f"Starting frame-by-frame analysis with frame skipping (every {skip_frames} frames)")
        
        # Generate synthetic coordinates
        start_lat, start_lon = generate_synthetic_coordinates()
        end_lat, end_lon = generate_synthetic_coordinates()
        logger.debug(f"Start location: {start_lat}, {start_lon}")
        logger.debug(f"End location: {end_lat}, {end_lon}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Unable to open video file")
            raise Exception("Unable to open video file")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        max_size = 640
        scale = min(max_size / width, max_size / height)
        if scale < 1:
            width, height = int(width * scale), int(height * scale)
        
        logger.debug(f"Video properties: {width}x{height}, {fps}fps")
        
        # Initialize video writer
        processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logger.error("Failed to initialize video writer")
            raise Exception("Failed to initialize video writer")
        
        frame_count = 0
        processed_frame_count = 0
        total_severity = 0
        unique_potholes = set()
        global_road_mask = np.zeros((height, width), dtype=np.uint8)
        global_pothole_mask = np.zeros((height, width), dtype=np.uint8)
        
        last_binary_mask = np.zeros((height, width), dtype=np.uint8)
        last_detected_potholes = []
        last_current_damage = 0
        
        saved_frame = False
        
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.debug("End of video reached")
                    break
                
                frame_count += 1
                if frame_count % (skip_frames * 10) == 0:
                    logger.info(f"Processed {frame_count} frames")

                if scale < 1:
                    frame = cv2.resize(frame, (width, height))
                output_frame = frame.copy()
                
                if frame_count % skip_frames == 0:
                    processed_frame_count += 1
                    
                    logger.debug(f"Running model prediction for frame {frame_count}")
                    results_list = model_predict(model, frame)
                    logger.debug(f"Model prediction completed for frame {frame_count}")
                    
                    if not results_list:
                        logger.debug(f"Frame {frame_count}: No potholes detected")
                        out.write(output_frame)
                        continue
                    
                    results = results_list[0]
                    binary_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    detected_potholes = []
                    
                    try:
                        if hasattr(results, 'boxes') and results.boxes is not None:
                            for box in results.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                confidence = float(box.conf[0])
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                pothole_id = (center_x // 10, center_y // 10)
                                if pothole_id not in unique_potholes:
                                    unique_potholes.add(pothole_id)
                                    logger.debug(f"Frame {frame_count}: New pothole at {pothole_id}")
                                detected_potholes.append((x1, y1, x2, y2, confidence))
                        
                        if hasattr(results, 'masks') and results.masks is not None and hasattr(results.masks, 'data'):
                            mask_data = results.masks.data.cpu().numpy()
                            for mask in mask_data:
                                mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                                binary_mask = cv2.bitwise_or(binary_mask, mask_resized)
                    except Exception as e:
                        logger.error(f"Error processing model results for frame {frame_count}: {str(e)}", exc_info=True)
                        continue
                    
                    logger.debug(f"Frame {frame_count}: Mask sum: {np.sum(binary_mask)}")
                    
                    try:
                        new_road_pixels = (frame[:, :, 0] > 50) | (frame[:, :, 1] > 50) | (frame[:, :, 2] > 50)
                        global_road_mask[new_road_pixels] = 255
                        global_pothole_mask = cv2.bitwise_or(global_pothole_mask, binary_mask)
                        
                        road_pixels = np.sum(new_road_pixels)
                        pothole_pixels = np.sum(binary_mask > 0)
                        current_damage = (pothole_pixels / road_pixels * 100) if road_pixels > 0 else 0
                        total_road_pixels = np.sum(global_road_mask > 0)
                        total_pothole_pixels = np.sum(global_pothole_mask > 0)
                        total_damage = (total_pothole_pixels / total_road_pixels * 100) if total_road_pixels > 0 else 0
                        
                        total_severity += current_damage
                        
                        last_binary_mask = binary_mask.copy()
                        last_detected_potholes = detected_potholes
                        last_current_damage = current_damage
                    except Exception as e:
                        logger.error(f"Error calculating damage for frame {frame_count}: {str(e)}", exc_info=True)
                        continue
                    
                    if not saved_frame and detected_potholes:
                        try:
                            temp_frame_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                            cv2.imwrite(temp_frame_path, output_frame)
                            saved_frame = True
                            logger.info(f"Saved frame for Telegram: {temp_frame_path}")
                        except Exception as e:
                            logger.error(f"Error saving frame for Telegram: {str(e)}", exc_info=True)
                
                else:
                    binary_mask = last_binary_mask
                    detected_potholes = last_detected_potholes
                    current_damage = last_current_damage
                    total_road_pixels = np.sum(global_road_mask > 0)
                    total_pothole_pixels = np.sum(global_pothole_mask > 0)
                    total_damage = (total_pothole_pixels / total_road_pixels * 100) if total_road_pixels > 0 else 0
                
                # Annotate and write frame
                try:
                    cv2.putText(
                        output_frame,
                        f"Total Damage: {(total_damage * 0.5):.2f}%",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        output_frame,
                        f"Current Frame: {current_damage:.2f}%",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2
                    )
                    cv2.putText(
                        output_frame,
                        f"Potholes: {len(detected_potholes)}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 2)
                    
                    mask_colored = np.zeros_like(output_frame)
                    mask_colored[:, :, 2] = binary_mask
                    output_frame = cv2.addWeighted(output_frame, 0.9, mask_colored, 0.1, 0)
                    
                    for x1, y1, x2, y2, confidence in detected_potholes:
                        cv2.putText(
                            output_frame,
                            f"{confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
                    
                    out.write(output_frame)
                except Exception as e:
                    logger.error(f"Error annotating/writing frame {frame_count}: {str(e)}", exc_info=True)
                    continue
            
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}", exc_info=True)
                continue
        
        avg_severity = total_severity / processed_frame_count if processed_frame_count > 0 else 0
        total_road_pixels = np.sum(global_road_mask > 0)
        total_pothole_pixels = np.sum(global_pothole_mask > 0)
        total_damage = (total_pothole_pixels / total_road_pixels * 100 * 0.5) if total_road_pixels > 0 else 0
        
        try:
            out.release()
            out = None
            logger.info("Video writer released")
        except Exception as e:
            logger.error(f"Error releasing video writer: {str(e)}", exc_info=True)
        
        try:
            logger.info("Validating processed video")
            validate_video_file(processed_video_path)
            logger.info(f"Processed video saved to: {processed_video_path}")
        except Exception as e:
            logger.error(f"Error validating processed video: {str(e)}", exc_info=True)
            raise
        
        try:
            logger.info("Uploading video to Cloudinary")
            video_url = upload_video_to_cloudinary(
                processed_video_path,
                public_id=f"pothole_detection/video_{uuid.uuid4()}"
            )
            logger.info(f"Uploaded processed video to: {video_url}")
        except Exception as e:
            logger.error(f"Error uploading to Cloudinary: {str(e)}", exc_info=True)
            video_url = "Upload failed"
        
        result = {
            "average_severity": float(round(avg_severity, 2)),
            "damaged_road_percentage": float(round(total_damage, 2)),
            "video_url": video_url,
            "start_location": {"lat": start_lat, "lon": start_lon},
            "end_location": {"lat": end_lat, "lon": end_lon}
        }
        
        logger.info(f"Final result: {result}")
        
        # SEND TELEGRAM MESSAGE
        report_message = (
            f" **Pothole Detection Report** \n"
            f"• Average severity: {result['average_severity']}%\n"
            f"• Damaged road percentage: {result['damaged_road_percentage']}%\n"
            f"• Start location: {result['start_location']['lat']}, {result['start_location']['lon']}\n"
            f"• End location: {result['end_location']['lat']}, {result['end_location']['lon']}\n"
            f"• Processed Video URL: {result['video_url']}\n"
        )
        
        logger.info("Preparing to send Telegram message")
        if temp_frame_path:
            logger.info(f"Sending Telegram message with photo: {temp_frame_path}")
        else:
            logger.info("Sending Telegram message without photo")
        
        try:
            asyncio.run(send_telegram_message_with_photo(report_message, photo_path=temp_frame_path))
            logger.info("Telegram message sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}", exc_info=True)
            raise
        
        return result
    
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}", exc_info=True)
        raise
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            logger.debug("Released video capture")
        if out is not None:
            out.release()
            logger.debug("Released video writer")
        if processed_video_path and os.path.exists(processed_video_path):
            try:
                os.remove(processed_video_path)
                logger.debug(f"Deleted temporary file: {processed_video_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")
        if temp_frame_path and os.path.exists(temp_frame_path):
            try:
                os.remove(temp_frame_path)
                logger.debug(f"Deleted temporary frame: {temp_frame_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary frame: {str(e)}")