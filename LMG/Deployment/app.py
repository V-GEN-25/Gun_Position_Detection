import streamlit as st
from gun_detection import GunDetectionSystem
import cv2
import tempfile
import numpy as np
from PIL import Image
import time
import pyngrok.ngrok as ngrok
import threading
import os

# Initialize the detection system
detection_system = GunDetectionSystem()

# Set up page config
st.set_page_config(
    page_title="Gun Position Detection",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .metric-value {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    .correct {
        color: #2ecc71 !important;
    }
    .incorrect {
        color: #e74c3c !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.title("🔫 Gun Position Detection")
    st.markdown("""
    This system analyzes proper gun handling posture by measuring the distance between 
    the gun butt and the shooter's shoulder.
    """)
    
    # Upload video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    # Demo selection
    demo_option = st.radio(
        "Or choose a demo video:",
        ("None", "Correct Posture", "Incorrect Posture")
    )
    
    # Threshold adjustment
    threshold = st.slider(
        "Distance Threshold (pixels)",
        min_value=50,
        max_value=150,
        value=90,
        help="Maximum allowed distance between gun butt and shoulder"
    )
    detection_system.threshold_distance = threshold
    
    # Performance stats
    stats = detection_system.get_performance_stats()
    st.metric("Average FPS", f"{stats['avg_fps']:.1f}")
    
    # Ngrok controls
    if st.checkbox("Enable Ngrok Public URL"):
        auth_token = st.text_input("Ngrok Auth Token", type="password")
        if st.button("Start Ngrok Tunnel"):
            if auth_token:
                ngrok.set_auth_token(auth_token)
                public_url = ngrok.connect(8501).public_url
                st.success(f"Public URL: {public_url}")
            else:
                st.error("Please enter your Ngrok auth token")

# Main content area
st.title("Gun Position Analysis")
st.markdown("""
Analyze proper gun handling posture in real-time or from uploaded videos.
""")

# Create columns for video and metrics
col1, col2 = st.columns([2, 1])

with col1:
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("Detection Metrics")
    distance_metric = st.metric("Distance (pixels)", "-")
    status_metric = st.metric("Position Status", "-")
    
    st.subheader("Recent Detections")
    history_placeholder = st.empty()

# Process video based on selection
if uploaded_file is not None or demo_option != "None":
    # Get the video file
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    else:
        video_path = "demo_correct.mp4" if demo_option == "Correct Posture" else "demo_incorrect.mp4"
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, results = detection_system.process_frame(frame)
        
        # Convert to RGB for Streamlit
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Update metrics
        if results["distance"] is not None:
            distance_metric.metric("Distance (pixels)", f"{results['distance']:.1f}")
            status_class = "correct" if results["status"] == "correct" else "incorrect"
            status_metric.metric(
                "Position Status", 
                results["status"].capitalize(),
                delta=None,
                delta_color="normal",
                help=None
            )
        
        # Display processed frame
        video_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
        
        # Update history
        stats = detection_system.get_performance_stats()
        if stats["detection_history"]:
            history_data = [
                {
                    "Time": time.strftime('%H:%M:%S', time.localtime(d["timestamp"])),
                    "Distance": f"{d['distance']:.1f}",
                    "Status": d["status"].capitalize()
                }
                for d in stats["detection_history"]
            ]
            history_placeholder.table(history_data[-5:])  # Show last 5 entries
        
        # Slow down to approximate original FPS
        time.sleep(1 / fps)
    
    cap.release()
    if uploaded_file is not None:
        os.unlink(video_path)
else:
    # Show placeholder when no video is selected
    video_placeholder.image(Image.open("placeholder.jpg"), use_column_width=True)
    status_placeholder.info("Please upload a video or select a demo option to begin analysis.")