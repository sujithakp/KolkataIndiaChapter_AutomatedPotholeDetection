# utils/visualization.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import config


def calc_damaged_area(results):
     
     # Initialize variables to hold total area and individual areas
    total_area = 0
    area_list = []

    # Perform operations if masks are available
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()   # Retrieve masks as numpy arrays
        image_area = masks.shape[1] * masks.shape[2]  # Calculate total number of pixels in the image
        for i, mask in enumerate(masks):
            binary_mask = (mask > 0).astype(np.uint8) * 255  # Convert mask to binary
            color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)  # Convert binary mask to color
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the binary mask
            contour = contours[0]  # Retrieve the first contour
            area = cv2.contourArea(contour)  # Calculate the area of the pothole
            area_list.append(area)  # Append area to the list
            cv2.drawContours(color_mask, [contour], -1, (0, 255, 0), 3)  # Draw the contour on the mask


    # Calculate and print areas after displaying the images
    for i, area in enumerate(area_list):
        #print(f"Area of Pothole {i+1}: {area} pixels")  
        total_area += area  # Sum the areas for total

    return total_area / image_area * 100  # Return the percentage of pothole area



def draw_detections(image, results):
    """
    Draw segments on the image
    
    Args:
        image: PIL Image object
        results: seg masks 
        
    
    Returns:
        PIL Image with masks drawn
    """
    result_image = results[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  
    
       
    return result_image
