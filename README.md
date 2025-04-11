# KolkataIndiaChapter_AutomatedPotholeDetection

## Project Overview
This project aims to detect and analyze potholes on roads using computer vision techniques. It was developed by the Kolkata India Chapter of Omdena and uses various deep learning approaches including object detection and instance segmentation to identify potholes and assess their severity.

## Problem Statement
Potholes are a significant hazard on roads that can cause vehicle damage, accidents, and traffic delays. Automated detection of potholes can help in:
- Quickly identifying road maintenance needs
- Prioritizing repairs based on severity
- Improving road safety for commuters
- Assisting municipal authorities in infrastructure management










## Approaches
The project explores multiple approaches to pothole detection:

### 1. Object Detection
- Faster R-CNN models using PyTorch Vision
- YOLOv11 medium for custom object detection
- Pre-trained models fine-tuned on pothole datasets

### 2. Instance Segmentation
- Mask R-CNN implementation for precise pothole boundaries
- YOLOv11 for instance segmentation with severity assessment
- Pixel-wise segmentation for damage quantification

## Performance Metrics
- mAP (mean Average Precision) as the primary evaluation metric
- Best model achieved 0.738 mAP using YOLOv11 Medium with Instance Segmentation

## Deployment
- Flask-based web application for model inference
- RESTful API for integration with other applications
- Potential PWA (Progressive Web App) implementation for mobile devices

## Dataset
- Curated from multiple sources including Kaggle and Mendeley
- Annotated using Roboflow and Microsoft Florence2
- Organized in COCO JSON format for training various models

## Tools Used
- Roboflow for annotation and dataset management
- Kaggle/Google Colab for model training
- PyTorch and YOLOv11 for model development
- Flask for web application development
- CodeSandBox.io for collaborative deployment













## Future Enhancements
- PWA implementation for cross-platform compatibility
- Real-time processing from video streams
- Integration with mapping services for geographic visualization
- Mobile application development


- ## Acknowledgments
This project was developed as part of the Omdena Kolkata India Chapter collaborative initiative. 
