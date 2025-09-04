# Add these imports to your main.py
import dlib
import cv2
import numpy as np
from scipy.spatial import distance
from typing import Tuple, List

# Add blink detection functionality
class BlinkDetector:
    def __init__(self):
        # Load face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        
        C = distance.euclidean(eye[0], eye[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blinks(self, video_path: str) -> Tuple[int, float]:
        EYE_AR_CONSEC_FRAMES = 2
        blink_counter = 0
        frame_counter = 0
        total_blinks = 0

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            return 0, 0.0

        (lStart, lEnd) = (42, 48)
        (rStart, rEnd) = (36, 42)

        # --- Calibration step ---
        EYE_AR_THRESH = 0.2
        baseline_ears = []
        for _ in range(min(5, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            if rects:
                shape = self.predictor(gray, rects[0])
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
                baseline_ears.append(ear)
        if baseline_ears:
            EYE_AR_THRESH = np.mean(baseline_ears) * 0.7

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind video

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)

            if len(rects) == 0:
                continue

            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0

                if ear < EYE_AR_THRESH:
                    frame_counter += 1
                else:
                    if frame_counter >= EYE_AR_CONSEC_FRAMES:
                        total_blinks += 1
                    frame_counter = 0

        video_duration = total_frames / fps
        blink_rate = (total_blinks / video_duration) * 60
        cap.release()
        return total_blinks, blink_rate


# Initialize blink detector
blink_detector = BlinkDetector()