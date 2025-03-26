# model/detector.py
import torch
import torchvision
from PIL import Image
import numpy as np
import os
#import cv2
import torch
import numpy as np
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR


class PotholeDetector:
    _instance = None
    
    @classmethod
    def get_instance(cls, model_path=None):
        """Singleton pattern to ensure model is loaded only once"""
        if cls._instance is None and model_path is not None:
            cls._instance = cls(model_path)
        return cls._instance
    
    def __init__(self, model_path):
        """Initialize the model - should be called only once"""
        print("Loading Faster R-CNN model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_classes = 5  # Adjust based on your dataset
        
        # Load the model 

        # Load a pre-trained ResNet50 backbone
        backbone = torchvision.models.resnet50(pretrained=True)
        # Remove the fully connected layer and average pooling layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048  # ResNet50 has 2048 output channels for the last convolutional layer

        # Anchor generator and ROI pooling
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

        # Faster R-CNN model with ResNet50 backbone
        self.model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        print("Model loaded successfully!")
    
    def detect(self, image):
        """Run inference on the image"""
        # Convert PIL Image to tensor
        if isinstance(image, Image.Image):
            image_tensor = torchvision.transforms.functional.to_tensor(image)
        else:
            # Handle numpy array
            image_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))) / 255.0
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Return the prediction for the first (and only) image
        return predictions[0]
