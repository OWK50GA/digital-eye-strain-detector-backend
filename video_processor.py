import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class VideoFeatureExtractor:
    def __init__(self):
        # Load ResNet model (same as used during training)
        self.resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool1d(512)
        self.resnet.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_features_from_video(self, video_path, target_seq_length=100):
        """
        Extract ResNet features from a video and prepare them for your model
        
        Args:
            video_path: Path to the video file
            target_seq_length: Number of sequences to extract (default: 100)
            
        Returns:
            features: List of 512-dimensional feature vectors
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame interval to get exactly target_seq_length frames
        frame_interval = max(1, total_frames // target_seq_length)
        
        features = []
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < target_seq_length:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract frame at regular intervals
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Apply transformations
                input_tensor = self.transform(pil_image)
                input_batch = input_tensor.unsqueeze(0)
                
                # Extract features
                with torch.no_grad():
                    feature = self.resnet(input_batch)
                    feature = self.adaptive_pool(feature.squeeze().unsqueeze(0))
                
                # Flatten and convert to numpy
                feature_vector = feature.squeeze().numpy()
                features.append(feature_vector)
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        
        # If we didn't get enough frames, pad with the last frame
        if len(features) < target_seq_length:
            last_feature = features[-1] if features else np.zeros(512)
            while len(features) < target_seq_length:
                features.append(last_feature)
        
        # If we got too many frames, truncate
        features = features[:target_seq_length]
        
        return features