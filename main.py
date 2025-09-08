import os
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, UploadFile, File
# from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
import time
from video_processor import VideoFeatureExtractor
import tempfile
# from blink_detector import BlinkDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define input and output schemas
class PredictionInput(BaseModel):
    data: List[List[float]]  # Shape: [SEQ_LEN, 512]

class PredictionOutput(BaseModel):
    prediction: int
    confidence: List[float]
    inference_time: float

# Corrected model architecture based on error analysis
class EyeStrainModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_classes=2):  # Changed hidden_size to 256
        super().__init__()
        
        # LSTM layer (bidirectional) - corrected dimensions
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # Now 256
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention pooling mechanism - corrected key names
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier layers - corrected input size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # Input should be 512 (256*2)
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (B, T, H*2)
        
        # Attention pooling
        attn_weights = self.attn_pool(lstm_out)  # (B, T, 1)
        attended_features = torch.sum(lstm_out * attn_weights, dim=1)  # (B, H*2)
        
        # Classification
        logits = self.classifier(attended_features)
        
        return logits

# Initialize FastAPI app
app = FastAPI(
    title="Eye Strain Detection API",
    description="API for detecting eye strain from video features",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",   # your frontend
    "http://127.0.0.1:3000",
    "https://your-frontend-domain.com",  # production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],          # ["GET", "POST", ...]
    allow_headers=["*"],
)

# Global model variable
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 100  # From your training code

@app.on_event("startup")
async def load_model():
    """Load the trained model during application startup"""
    global model
    try:
        # Initialize model with the correct architecture
        model = EyeStrainModel(
            input_size=512, 
            hidden_size=256,  # Changed to 256 based on error analysis
            num_classes=2
        ).to(device)
        
        # Load model weights
        model_path = "resnet_lstm_best_with_threshold.pth"
        checkpoint = torch.load(model_path, map_location=device)
        
        # Debug: print keys from checkpoint and model
        logger.info("Keys in checkpoint:")
        for key in checkpoint['model_state_dict'].keys():
            logger.info(f"  {key}")
            
        logger.info("Keys in model:")
        for key in model.state_dict().keys():
            logger.info(f"  {key}")
        
        # Load state dict with strict=False to skip mismatched keys
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        logger.info(f"Model loaded successfully on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

# ... rest of your FastAPI code remains the same ...

# @app.post("/process_video")
# async def process_video(video_file: UploadFile = File(...)):
#     """
#     Process a video file and return features ready for prediction
#     """
#     try:
#         # Save uploaded video temporarily
#         temp_video_path = f"temp_{video_file.filename}"
#         with open(temp_video_path, "wb") as buffer:
#             content = await video_file.read()
#             buffer.write(content)
        
#         # Extract features
#         extractor = VideoFeatureExtractor()
#         features = extractor.extract_features_from_video(temp_video_path)
        
#         # Clean up
#         os.remove(temp_video_path)
        
#         return {"features": features, "message": "Video processed successfully"}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Eye Strain Detection API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/process_video")
async def process_video(video_file: UploadFile = File(...)):
    """
    Process a video file and return features ready for prediction
    """
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await video_file.read()
            temp_file.write(content)
            temp_video_path = temp_file.name
        
        # Extract features
        extractor = VideoFeatureExtractor()
        # blink_detector = BlinkDetector()
        features = extractor.extract_features_from_video(temp_video_path)
        
        # blink_count, blink_rate = blink_detector.detect_blinks(temp_video_path)
        features_list = [feature.tolist() for feature in features]
        
        # Clean up
        os.unlink(temp_video_path)
        
        return {
            "features": features_list,
            # "blink_count": blink_count,
            # "blink_rate": blink_rate,
            "message": "Video processed successfully"
        }
        
    except Exception as e:
        # Ensure cleanup even if error occurs
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make predictions using the loaded PyTorch model
    
    Expected input: 2D array of shape [100, 512] (SEQ_LEN x feature_size)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Convert input to numpy array and check shape
        input_array = np.array(input_data.data, dtype=np.float32)
        
        if input_array.shape != (SEQ_LEN, 512):
            raise HTTPException(
                status_code=400, 
                detail=f"Input shape must be ({SEQ_LEN}, 512). Got {input_array.shape}"
            )
        
        # Apply the same preprocessing as in training
        # L2 normalize per frame
        norms = np.linalg.norm(input_array, axis=1, keepdims=True)
        input_array = input_array / (norms + 1e-8)
        
        # Global standardization across the sequence
        input_array = (input_array - np.mean(input_array)) / (np.std(input_array) + 1e-8)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        return PredictionOutput(
            prediction=prediction.item(),
            confidence=probabilities.cpu().numpy()[0].tolist(),
            inference_time=inference_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_architecture": str(model),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "input_shape": f"(1, {SEQ_LEN}, 512)",
        "output_classes": 2,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)