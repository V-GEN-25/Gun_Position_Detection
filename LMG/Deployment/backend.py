from flask import Flask, request, Response
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import torch
import threading
import run_ngrok
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gun = YOLO("best.pt").to(device)
human = YOLO("yolov8m-pose.pt").to(device)

def process_frame(frame):
    """Process single frame with gun/human detection"""
    lmg_results = gun(frame)
    pose_results = human(frame)
    
    # Gun detection
    if not lmg_results[0].boxes:
        return frame, None
    
    # Get best gun detection
    best_idx = torch.argmax(lmg_results[0].boxes.conf).item()
    gun_kpts = lmg_results[0].keypoints.xy[best_idx].cpu().numpy()
    
    # Human detection
    if not pose_results[0].keypoints:
        return frame, None
    
    # Calculate distance (example: gun butt to right shoulder)
    butt = gun_kpts[0]  # Adjust index per your model
    shoulder = pose_results[0].keypoints.xy[0][6].cpu().numpy()  # Right shoulder
    
    distance = np.linalg.norm(butt - shoulder)
    is_proper = distance <= 90
    
    # Annotate frame
    annotated = frame.copy()
    cv2.line(annotated, 
             tuple(map(int, butt)), 
             tuple(map(int, shoulder)), 
             (0, 0, 255), 2)
    cv2.putText(annotated, f"Dist: {distance:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated, is_proper

@app.route('/process-video', methods=['POST'])
def handle_video():
    if 'file' not in request.files:
        return {"error": "No video file"}, 400
    
    # Save uploaded file
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.mp4")
    request.files['file'].save(input_path)
    
    # Video reader/writer setup
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = os.path.join(temp_dir, "output.mp4")
    out = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, 
                         (width, height))
    
    # Process frames
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, is_proper = process_frame(frame)
        out.write(processed_frame)
        if is_proper is not None:
            results.append(is_proper)
    
    cap.release()
    out.release()
    
    # Return video file
    with open(output_path, 'rb') as f:
        video_bytes = f.read()
    
    # Cleanup
    os.remove(input_path)
    os.remove(output_path)
    os.rmdir(temp_dir)
    
    return Response(
        video_bytes,
        mimetype='video/mp4',
        headers={
            'X-Proper-Ratio': str(np.mean(results) if results else 0),
            'Content-Disposition': 'attachment; filename=processed.mp4'
        }
    )

def start_ngrok():
    ngrok_token = os.getenv("NGROK_AUTHTOKEN")
    listener = run_ngrok.forward(5000, authtoken=ngrok_token)
    print(f"🔗 Ngrok URL: {listener.url()}")

if __name__ == '__main__':
    threading.Thread(target=start_ngrok, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)