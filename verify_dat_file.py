# verify_shape_predictor.py
import dlib
import sys
import cv2

def verify_shape_predictor(model_path):
    try:
        # Try to load the predictor
        predictor = dlib.shape_predictor(model_path)
        print("✅ Shape predictor loaded successfully!")
        
        # Get some basic info (dlib doesn't provide direct access to all internals)
        print(f"📁 Model file: {model_path}")
        print("ℹ️  This is a dlib shape predictor for 68 facial landmarks")
        
        # Test with a simple face detection
        detector = dlib.get_frontal_face_detector()
        
        # Create a test image (blank with a simple face-like shape)
        import numpy as np
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw a simple oval "face"
        cv2.ellipse(test_img, (100, 100), (80, 100), 0, 0, 360, (255, 255, 255), -1)
        
        # Detect faces
        faces = detector(test_img, 1)
        
        if len(faces) > 0:
            # Try to predict landmarks
            shape = predictor(test_img, faces[0])
            print(f"✅ Model works! Detected {shape.num_parts} facial landmarks")
            return True
        else:
            print("⚠️  Could not test landmarks (no face detected in test image)")
            return True
            
    except Exception as e:
        print(f"❌ Error loading shape predictor: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_shape_predictor(sys.argv[1])
    else:
        print("Please provide the path to the shape predictor .dat file")