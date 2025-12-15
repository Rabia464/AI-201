"""
Simple rule-based emotion "recognizer" (no CNN, no training).
Uses trivial image statistics on the detected face to choose between:
Happy, Sad, Energetic, Calm.
"""

import cv2
import numpy as np


class EmotionRecognizer:
    """
    Purely rule-based "emotion recognizer".
    No neural networks, no training – just simple image statistics.
    """

    def __init__(self, model_path=None, device=None):
        # model_path and device are ignored; kept only for compatibility.
        self.emotion_labels = ['Happy', 'Sad', 'Energetic', 'Calm']
        self.last_features = {}
    
    def predict(self, face_image):
        """
        Predict emotion from face image using simple handcrafted rules.
        """
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image

            # Normalize to a fixed size for consistent statistics
            face_resized = cv2.resize(gray, (200, 200))

            mean_intensity = np.mean(face_resized)
            std_intensity = np.std(face_resized)

            h, w = face_resized.shape
            mouth_region = face_resized[int(h * 0.6):, :]
            eye_region = face_resized[:int(h * 0.4), :]

            mouth_mean = np.mean(mouth_region)
            eye_mean = np.mean(eye_region)
            mouth_std = np.std(mouth_region)

            edges = cv2.Canny(face_resized, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)

            # Save features for external inspection
            self.last_features = {
                "mean_intensity": round(float(mean_intensity), 2),
                "std_intensity": round(float(std_intensity), 2),
                "mouth_mean": round(float(mouth_mean), 2),
                "mouth_std": round(float(mouth_std), 2),
                "eye_mean": round(float(eye_mean), 2),
                "edge_density": round(float(edge_density), 4),
            }

            # Continuous scoring so every frame can map to one of 4 emotions.
            # These are simple formulas, not ML.
            happy_score = max(0.0,
                              (mouth_mean - mean_intensity) / 15.0 +
                              1.5 * edge_density)

            sad_score = max(0.0,
                            (mean_intensity - mouth_mean) / 15.0 +
                            0.2 * max(0.0, 0.04 - edge_density))

            energetic_score = max(0.0,
                                  std_intensity / 40.0 +
                                  2.0 * edge_density)

            calm_score = max(0.0,
                             max(0.0, 25.0 - std_intensity) / 25.0 +
                             max(0.0, 0.05 - edge_density))

            emotion_scores = {
                'Happy': float(happy_score),
                'Sad': float(sad_score),
                'Energetic': float(energetic_score),
                'Calm': float(calm_score),
            }

            max_score = max(emotion_scores.values())
            sum_scores = sum(emotion_scores.values())

            # If nothing is convincing, fall back to Calm with moderate confidence
            if max_score <= 0.0 or sum_scores <= 0.0:
                return "Calm", 0.5

            predicted_emotion = max(emotion_scores, key=emotion_scores.get)
            # Confidence = contribution of the winning score over total scores (0.5–0.95)
            raw_conf = max_score / (sum_scores + 1e-6)
            confidence = 0.5 + 0.45 * float(np.clip(raw_conf, 0.0, 1.0))

            return predicted_emotion, confidence

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Fallback to brightness-only heuristic
            try:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
                brightness = np.mean(gray)
                if brightness > 120:
                    return "Happy", 0.7
                elif brightness < 80:
                    return "Sad", 0.7
                else:
                    return "Calm", 0.7
            except:
                return "Calm", 0.6

    def get_last_features(self):
        """Return the last computed feature dictionary (for debugging/UI)."""
        return self.last_features

