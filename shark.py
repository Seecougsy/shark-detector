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
import torch
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import cv2
import tempfile
import os
import time
import numpy

import streamlit as st
from ultralytics import solutions


# Inject custom CSS
with open("Static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
model = YOLO("best.pt")
model_live = YOLO("best_nano.pt")

model_live.model.fuse()  # fuse Conv+BN layers

import torch.nn as nn, torch

model_live.model = torch.quantization.quantize_dynamic(
    model_live.model, {nn.Conv2d}, dtype=torch.qint8
)

st.markdown(
    """
<div class="title-box">
    <div class="custom-title">Safe Distance Shark Monitor 🚁</div>
    <div class="subtitle">Using YOLOv8 for real-time shark detection</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """ <div class="section-box"> 
            <div class="section-title">1. Set Detection Confidence Threshold</div>
            <div class="section-subtitle">Only predictions above this confidence level will be shown.</div>
            </div>""",
    unsafe_allow_html=True,
)
conf_threshold = st.slider(
    "Confidence threshold",
    0.2,
    1.0,
    0.7,
    0.05,
    help="Drag to set the minimum detection confidence",
)



# ========== IMAGE DETECTION ==========
st.info(" ## Select a Detection Mode"
" ## Upload an image or a video to detect sharks")
st.markdown(
    """
<div class="section-box">
    <div class="section-title">
        2
    </div>
    <div class="section-subtitle">Upload an image or a video to detect sharks</div>
</div>
""",
    unsafe_allow_html=True,
)
mode = st.selectbox("Choose Mode", ["Select...", "Image", "Video", "Live"])

if mode == "Image":
    with st.container():
        st.subheader("🖼️ Image Detection")
        uploaded_img = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"], key="image"
        )

        if uploaded_img is not None:
            image = Image.open(uploaded_img)
            results = model.predict(image, conf=conf_threshold, iou=0.5)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated, caption="Detection result", use_container_width=True)

            detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
            if "shark" in detected_classes:
                st.error("Shark detected in image.")
            else:
                st.info("No shark detected.")

elif mode == "Video":
    with st.container():
        st.subheader("🎥 Video Detection")
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"], key="video")
        if uploaded_video:
            # write temp file
            tfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            stframe = st.empty()
            model_live.to("cuda" if torch.cuda.is_available() else "cpu")
            torch.set_grad_enabled(False)

            for result in model_live.predict(
                source=video_path,
                stream=True,
                conf=conf_threshold,
                vid_stride=4,
                augment=False,
                show_conf=False,
                show_labels=False,
            ):
                # get the raw frame back
                frame = result.orig_img.copy()

                # draw each box (no text)
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                stframe.image(
                    frame,
                    channels="BGR",
                    use_container_width=True,
                    output_format="JPEG",
                )

            tfile.close()


elif mode == "Live":
    with st.container():
        st.info("""
            ## Stream Webcam  
            Click **Start** to begin live detection via your webcam.
            """)

        # Initialize webcam state
        if "webcam_on" not in st.session_state:
            st.session_state.webcam_on = False

        start = st.button("Start")
        stop = st.button("Stop")
        if start:
            st.session_state.webcam_on = True
        if stop:
            st.session_state.webcam_on = False

        st.markdown(
            f"**Status:** {'Running' if st.session_state.webcam_on else 'Stopped'}"
        )

        if st.session_state.webcam_on:
            stframe = st.empty()
            model_live.to("cuda" if torch.cuda.is_available() else "cpu")
            torch.set_grad_enabled(False)

            for result in model_live.predict(
                source=0,  # 0 = default webcam
                stream=True,  # frame-by-frame generator
                conf=conf_threshold,
                iou=0.5,
                vid_stride=2,  # process every 3rd frame
                show_conf=False,
                show_labels=False,
                augment=False,
            ):
                frame = result.plot()
                stframe.image(
                    frame,
                    channels="BGR",
                    output_format="JPEG",
                    use_container_width=True,
                )
