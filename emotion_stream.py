"""
MVP Emotion Recognition Stream
Author: Modified for real-time emotion output
Description: Continuously outputs emotion data in JSON format
"""
import sys
import time
import json
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms.transforms as transforms
from face_detector.face_detector import DnnDetector, HaarCascadeDetector
from model.model import Mini_Xception
from utils import get_label_emotion, histogram_equalization
from face_alignment.face_alignment import FaceAlignment

sys.path.insert(1, 'face_detector')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmotionStream:
    def __init__(self, use_haar=False, pretrained_path='checkpoint/model_weights/weights_epoch_75.pth.tar'):
        # Initialize model
        self.mini_xception = Mini_Xception().to(device)
        self.mini_xception.eval()
        
        # Load model weights
        checkpoint = torch.load(pretrained_path, map_location=device)
        self.mini_xception.load_state_dict(checkpoint['mini_xception'])
        
        # Initialize face detection and alignment
        self.face_alignment = FaceAlignment()
        root = 'face_detector'
        
        if use_haar:
            self.face_detector = HaarCascadeDetector(root)
        else:
            self.face_detector = DnnDetector(root)
        
        # Initialize camera
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise RuntimeError("Could not open camera")
        
        print("Emotion recognition system initialized successfully!")
        print("Starting real-time emotion detection...")
        print("Press Ctrl+C to stop")
    
    def detect_emotion(self, frame):
        """Detect emotion from a single frame"""
        faces = self.face_detector.detect_faces(frame)
        
        if not faces:
            return None
        
        # Process the first detected face
        face = faces[0]
        (x, y, w, h) = face
        
        # Preprocess face
        input_face = self.face_alignment.frontalize_face(face, frame)
        input_face = cv2.resize(input_face, (48, 48))
        input_face = histogram_equalization(input_face)
        
        # Convert to tensor
        input_face = transforms.ToTensor()(input_face).to(device)
        input_face = torch.unsqueeze(input_face, 0)
        
        # Predict emotion
        with torch.no_grad():
            emotion_logits = self.mini_xception(input_face)
            
            # Get probabilities
            softmax = torch.nn.Softmax(dim=0)
            emotions_soft = softmax(emotion_logits.squeeze()).cpu().detach().numpy()
            
            # Get predicted emotion
            predicted_emotion = torch.argmax(emotion_logits)
            confidence = round(emotions_soft[predicted_emotion].item(), 3)
            emotion_label = get_label_emotion(predicted_emotion.squeeze().cpu().item())
            
            # Create emotion probabilities dictionary
            emotion_probs = {}
            for i, prob in enumerate(emotions_soft):
                emotion_probs[get_label_emotion(i)] = round(prob.item(), 3)
            
            return {
                "emotion": emotion_label,
                "confidence": confidence,
                "probabilities": emotion_probs,
                "face_detected": True,
                "face_position": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            }
    
    def stream_emotions(self, output_format='json', show_video=False):
        """Continuously stream emotion data"""
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    break
                
                # Detect emotion
                emotion_data = self.detect_emotion(frame)
                
                if emotion_data:
                    emotion_data["timestamp"] = time.time()
                    emotion_data["frame_count"] = frame_count
                    
                    if output_format == 'json':
                        print(json.dumps(emotion_data))
                    elif output_format == 'simple':
                        print(f"Emotion: {emotion_data['emotion']} (Confidence: {emotion_data['confidence']})")
                
                # Optional: show video window
                if show_video:
                    if emotion_data:
                        # Draw emotion on frame
                        cv2.putText(frame, emotion_data['emotion'], (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {emotion_data['confidence']}", (10, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Emotion Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                
                # Calculate and display FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    if output_format == 'simple':
                        print(f"FPS: {fps:.1f}")
                
        except KeyboardInterrupt:
            print("\nStopping emotion detection...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.video.release()
        cv2.destroyAllWindows()
        print("Emotion detection stopped.")

def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition Stream')
    parser.add_argument('--haar', action='store_true', help='Use Haar cascade face detector')
    parser.add_argument('--pretrained', type=str, default='checkpoint/model_weights/weights_epoch_75.pth.tar',
                       help='Path to model weights')
    parser.add_argument('--format', type=str, choices=['json', 'simple'], default='json',
                       help='Output format: json or simple')
    parser.add_argument('--show-video', action='store_true', help='Show video window')
    
    args = parser.parse_args()
    
    try:
        emotion_stream = EmotionStream(use_haar=args.haar, pretrained_path=args.pretrained)
        emotion_stream.stream_emotions(output_format=args.format, show_video=args.show_video)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
