import cv2
import numpy as np
from ultralytics import YOLO
import torch
import tempfile
import time
from collections import deque

class GunDetectionSystem:
    def __init__(self):
        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gun_model = YOLO("best.pt").to(self.device)
        self.human_model = YOLO("yolov8m-pose.pt").to(self.device)
        
        # Constants
        self.lmg_labels = ['butt', 'piston grip', 'trigger', 'cover', 'rear sight', 
                          'barrel jacket', 'left bipod', 'right bipod']
        self.pose_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                           "left_wrist", "right_wrist", "left_hip", "right_hip",
                           "left_knee", "right_knee", "left_ankle", "right_ankle"]
        
        # Performance tracking
        self.frame_times = deque(maxlen=20)
        self.detection_history = []
        self.threshold_distance = 90  # Distance threshold for good posture
    
    def process_frame(self, frame):
        start_time = time.time()
        
        # Run inference
        with torch.no_grad():
            lmg_result = self.gun_model(frame)
            pose_result = self.human_model(frame)
        
        # Process gun detection
        confidence_scores = []
        for conf in lmg_result[0].boxes:
            confidence_scores.append(float(conf.conf[0]))
        
        if confidence_scores:
            index = confidence_scores.index(max(confidence_scores))
            keypoints = lmg_result[0].keypoints.xy[index]
            lmg_coords = {self.lmg_labels[i]: tuple(keypoints[i].cpu().squeeze().numpy()) 
                         for i in range(len(self.lmg_labels))}
        else:
            return frame, {"status": "no_gun", "distance": None}
        
        # Process human pose
        persons = [pose_result[0].keypoints.xy[i] 
                  for i in range(len(pose_result[0].keypoints.xy))]
        
        if not persons:
            return frame, {"status": "no_person", "distance": None}
        
        # For simplicity, we'll use the first detected person
        person = persons[0]
        if len(person) != 17:  # Ensure all keypoints are present
            return frame, {"status": "incomplete_pose", "distance": None}
        
        pose_coords = {self.pose_labels[i]: tuple(person[i].cpu().squeeze().numpy())
                      for i in range(len(self.pose_labels))}
        
        # Calculate distance between gun butt and right shoulder
        if "butt" in lmg_coords and "right_shoulder" in pose_coords:
            butt = tuple(map(int, lmg_coords["butt"]))
            right_shoulder = tuple(map(int, pose_coords["right_shoulder"]))
            
            distance = np.sqrt((right_shoulder[0] - butt[0]) ** 2 + 
                             (right_shoulder[1] - butt[1]) ** 2)
            
            # Visualize
            cv2.line(frame, butt, right_shoulder, (0, 0, 255), 2)
            cv2.circle(frame, butt, 5, (0, 255, 0), -1)
            cv2.circle(frame, right_shoulder, 5, (0, 0, 255), -1)
            
            # Determine status
            status = "correct" if distance <= self.threshold_distance else "incorrect"
            
            # Add text to frame
            color = (0, 255, 0) if status == "correct" else (0, 0, 255)
            text = "Correct Position" if status == "correct" else "Incorrect Position"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)
            
            # Track performance
            self.frame_times.append(time.time() - start_time)
            self.detection_history.append({
                "distance": distance,
                "status": status,
                "timestamp": time.time()
            })
            
            return frame, {"status": status, "distance": distance}
        
        return frame, {"status": "no_keypoints", "distance": None}
    
    def get_performance_stats(self):
        if not self.frame_times:
            return {"avg_fps": 0, "detection_history": []}
        
        avg_fps = 1 / (sum(self.frame_times) / len(self.frame_times))
        return {
            "avg_fps": avg_fps,
            "detection_history": self.detection_history[-20:]  # Last 20 detections
        }