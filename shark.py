# -----------------------------------------------------------------------------
# YOLOv8 Shark Detector
#
# This project uses Ultralytics YOLOv8 (AGPL-3.0) by Glenn Jocher et al.
# https://github.com/ultralytics/ultralytics
#
# By using or distributing this code, you agree to the terms of the
# GNU Affero General Public License v3.0:
# https://www.gnu.org/licenses/agpl-3.0.txt
# -----------------------------------------------------------------------------


# Import necessary libraries
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import cv2
import tempfile
import os

# Inject custom CSS
with open("Static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
model = YOLO("best.pt")

st.markdown("""
<div class="title-box">
    <div class="custom-title">Safe Distance Shark Monitor 🚁</div>
    <div class="subtitle">Using YOLOv8 for real-time shark detection</div>
</div>
""", unsafe_allow_html=True)

st.markdown(""" <div class="section-box"> 
            <div class="section-title">1. Set Detection Confidence Threshold</div>
            <div class="section-subtitle">Only predictions above this confidence level will be shown.</div>
            </div>""", unsafe_allow_html=True)
conf_threshold = st.slider("Confidence threshold", 
                           0.2, 1.0, 0.7, 0.05,
                           help="Drag to set the minimum detection confidence"
                           )


# ========== IMAGE DETECTION ==========
st.markdown("""
<div class="section-box">
    <div class="section-title">
        2. Select a Detection Mode
    </div>
    <div class="section-subtitle">Upload an image or a video to detect sharks</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="mini-box">
        <div class="mini-title">🖼️ Image Detection</div>
        <div class="mini-subtitle">Find a shark in an image</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image")


if uploaded_img is not None:
    image = Image.open(uploaded_img)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict with custom confidence
    results = model.predict(image, conf=conf_threshold, iou=0.5, save=True, project="runs", name="shark-detect", exist_ok=True)
    save_dir = results[0].save_dir
    saved_img_path = next((os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))), None)

    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]

    if "shark" in detected_classes:
        st.error("Shark detected in image.")
    else:
        st.info("No shark detected.")
    st.image(saved_img_path, caption="Detection Result (Saved)", use_container_width=True)

    st.success(f"✅ Detection result saved to: {saved_img_path}")
    

# ========== VIDEO DETECTION ==========
with col2:
    st.markdown("""
    <div class="mini-box">
        <div class="mini-title">🎥 Video Detection</div>
        <div class="mini-subtitle">Find a shark in a video</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a video", type=["mp4"], key="video")

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.video(video_path)
    st.info("Processing video, please wait...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = os.path.join("runs", "shark-detect-video")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Use selected confidence threshold for prediction
        results = model.predict(frame, conf=conf_threshold, iou=0.5)
        frame = results[0].plot()
        out.write(frame)

    cap.release()
    out.release()

    st.success("✅ Video detection complete!")
    st.video(out_path)
    with open(out_path, "rb") as file:
        st.download_button(label="📥 Download Result Video", data=file, file_name="shark_detected.mp4", mime="video/mp4")
