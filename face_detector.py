"""
Face Detection and Emotion Recognition Module
Uses OpenCV for face detection and integrates with emotion recognition model
"""

import cv2
import numpy as np
from emotion_model import EmotionRecognizer


class FaceDetector:
    """
    Handles real-time face detection using OpenCV.
    """
    
    def __init__(self):
        """Initialize face detector with OpenCV's Haar Cascade."""
        try:
            # Try to load Haar Cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Failed to load Haar Cascade classifier")
                
        except Exception as e:
            print(f"Error initializing face detector: {e}")
            self.face_cascade = None
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image frame from camera
            
        Returns:
            list: List of (x, y, w, h) bounding boxes for detected faces
        """
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            # OpenCV may return a NumPy array or a tuple; convert safely to list of boxes
            try:
                return list(faces)
            except TypeError:
                return []
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def extract_face(self, frame, face_box):
        """
        Extract and return face region from frame.
        
        Args:
            frame: BGR image frame
            face_box: (x, y, w, h) bounding box
            
        Returns:
            NumPy array: Cropped face image
        """
        try:
            x, y, w, h = face_box
            # Add some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            face = frame[y:y+h, x:x+w]
            return face
        except Exception as e:
            print(f"Error extracting face: {e}")
            return None


class EmotionDetectionSystem:
    """
    Complete system for face detection and emotion recognition.
    Combines FaceDetector and EmotionRecognizer.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion detection system.
        
        Args:
            model_path: Path to emotion recognition model (optional)
        """
        self.face_detector = FaceDetector()
        self.emotion_recognizer = EmotionRecognizer(model_path=model_path)
        self.current_emotion = "Unknown"
        self.current_confidence = 0.0
        self.last_features = {}
    
    def process_frame(self, frame):
        """
        Process a single frame: detect face and recognize emotion.
        
        Args:
            frame: BGR image frame from camera
            
        Returns:
            tuple: (processed_frame, emotion, confidence, face_box)
        """
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            if len(faces) == 0:
                self.current_emotion = "No Face Detected"
                self.current_confidence = 0.0
                return frame, "No Face Detected", 0.0, None
            
            # Use the first detected face
            face_box = faces[0]
            x, y, w, h = face_box
            
            # Extract face
            face_image = self.face_detector.extract_face(frame, face_box)
            
            if face_image is None:
                return frame, "Unknown", 0.0, face_box
            
            # Recognize emotion
            emotion, confidence = self.emotion_recognizer.predict(face_image)
            
            self.current_emotion = emotion
            self.current_confidence = confidence
            self.last_features = self.emotion_recognizer.get_last_features()
            
            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return frame, emotion, confidence, face_box
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, "Error", 0.0, None
    
    def get_current_emotion(self):
        """Get the most recently detected emotion."""
        return self.current_emotion, self.current_confidence

    def get_last_features(self):
        """Expose the last computed features for debugging or display."""
        return self.last_features

