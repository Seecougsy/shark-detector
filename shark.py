# YOLOv8 Shark Detector
#
# This project uses Ultralytics YOLOv8 (AGPL-3.0) by Glenn Jocher et al.
# https://github.com/ultralytics/ultralytics
#
# By using or distributing this code, you agree to the terms of the
# GNU Affero General Public License v3.0:
# https://www.gnu.org/licenses/agpl-3.0.txt

import streamlit as st
import tempfile
import os
from PIL import Image
import cv2
import torch
from ultralytics import YOLO

# WebRTC for browser-based webcam streaming
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# Configure page layout

# === User-configurable model paths ===
# Set your custom YOLO weight file paths here:
MODEL_PATH_FILE = "best.pt"        # e.g. your fine-tuned file detector
MODEL_PATH_STREAM = "best_nano.pt" # e.g. smaller model for live streaming

# Cache and load models only once per session
@st.cache_resource
def load_models(file_model_path: str, stream_model_path: str):
    # Load file-based model
    file_model = YOLO(file_model_path)
    file_model.fuse()
    # Load live-stream model
    stream_model = YOLO(stream_model_path)
    stream_model.fuse()
    return file_model, stream_model

# Initialize models
model, model_live = load_models(MODEL_PATH_FILE, MODEL_PATH_STREAM)

# App title
st.title("🚁 Safe Distance Shark Monitor")
st.subheader("Real‑time & file‑based shark detection with YOLOv8 and WebRTC")

# Step 1: Confidence threshold
st.header("1. Set Detection Confidence")
conf_threshold = st.slider(
    "Detection confidence (0–1)", 0.0, 1.0, 0.7, 0.05,
    help="Higher → fewer false positives; lower → more sensitivity"
)

# Step 2: Media source selection
st.header("2. Choose Input Method")
tab_upload, tab_live = st.tabs(["📁 Upload File", "📹 Live Webcam"])

# Prepare output area
output_slot = st.container()
frame_slot = output_slot.empty()

# Uploaded file branch
with tab_upload:
    st.info("Upload an image (jpg/png) or video (mp4) for shark detection.")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "mp4"], key="file")
    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(uploaded_file)
            results = model.predict(img, conf=conf_threshold, iou=0.5)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_slot.image(annotated_rgb, use_container_width=True)
            labels = [model.names[int(b.cls)] for b in results[0].boxes]
            if "shark" in labels:
                st.success("🦈 Shark detected!")
            else:
                st.info("✔️ No shark detected.")
        else:
            # Handle video upload stream
            tfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tfile.write(uploaded_file.read())
            tfile_path = tfile.name
            tfile.close()
            for res in model_live.predict(
                source=tfile_path, stream=True,
                conf=conf_threshold, vid_stride=4, iou=0.5
            ):
                frame = res.orig_img.copy()
                for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                frame_slot.image(frame, channels="BGR", use_container_width=True)
            os.remove(tfile_path)

# Live webcam branch using WebRTC
class SharkTransformer(VideoTransformerBase):
    def __init__(self):
        # initialize alert flag
        self.alerted = False
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # run detection
        results = model_live(img, conf=conf_threshold, iou=0.5)
        annotated = results[0].plot()
        # send toast on first detection
        names = [model_live.names[int(b.cls)] for b in results[0].boxes]
        if "shark" in names and not self.alerted:
            st.toast("🦈 Shark spotted live!", icon="🦈")
            self.alerted = True
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

with tab_live:
    st.info("Allow webcam access to simulate a live beach monitoring system.")
    webrtc_streamer(
        key="shark-webrtc", 
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SharkTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
