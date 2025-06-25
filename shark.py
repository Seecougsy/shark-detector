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

# st.markdown(
#     """ <div class="section-box">
#             <div class="section-title">1. Set Detection Confidence Threshold</div>
#             <div class="section-subtitle">Only predictions above this confidence level will be shown.</div>
#             </div>""",
#     unsafe_allow_html=True,
# )
st.header("Detection Process")

st.markdown("Using :blue-background[YOLOv8] for :rainbow[***real-time***] shark detection, the system processes :blue[***images***], :orange[***videos***], and :green[***live feeds***] to demonstrate the potential of computer vision technology in real-world applications." , width ="content")

with st.container():
    st.subheader("1. Set Confidence Threshold")
    st.markdown("- A :red-background[higher:material/stat_1:] confidence threshold will result in ***fewer***, but more ***certain***, detections." )
    st.markdown("- A :blue-background[lower:material/stat_minus_1:] confidence threshold may capture more subtle detections but may ***also*** include more ***false alarms.***", )
    conf_threshold = st.slider(
        "Confidence threshold",
        0.2,
        1.0,
        0.7,
        0.05,
        help="Drag to set the minimum detection confidence",
    )

    st.subheader("2. Upload or Stream Media")
    # Controls in columns
    col1, col3 = st.tabs([":material/upload: Upload File", ":material/camera_video: Live Stream",])

    with col1:
        st.markdown("Upload an image or video to see detections on recorded media.")
        uploaded_file = st.file_uploader("", type=["jpg","png","mp4"])
    with col3:
        st.markdown("Quick and dirty way to simulate a live environment for beach monitoring — just point your webcam at drone footage on another device")
        st.info("Toggle to start/stop webcam stream.")
        stream_on = st.checkbox("Activate Webcam", value=False, key="webcam_on")

    # Full-width output container
    output_slot = st.container()
    frame_slot = output_slot.empty()

    # Handle uploaded file

    if uploaded_file:
        st.session_state.shark_alerted = False
        filename = uploaded_file.name.lower()
        if filename.endswith(".mp4"):
            # video

   

            tfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tfile.write(uploaded_file.read())
            tfile.close()

            model_live.to("cuda" if torch.cuda.is_available() else "cpu")

            for res in model_live.predict(
                source=tfile.name,
                stream=True,
                conf=conf_threshold,
                vid_stride=4,
                iou=0.5,
            ):
                frame = res.orig_img.copy()
                for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )
                frame_slot.image(frame, channels="BGR", use_container_width=True)


                            # only toast the first time we actually detect a shark
                names = [model.names[int(b.cls)] for b in res.boxes]
                if "shark" in names and not st.session_state.shark_alerted:
                    st.toast("The model detected a Shark 🦈")
                    st.session_state.shark_alerted = True
            os.remove(tfile.name)


        else:
            # image
            img = Image.open(uploaded_file)
            results = model.predict(img, conf=conf_threshold, iou=0.5)
            ann = results[0].plot()
            ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            frame_slot.image(ann_rgb, caption="Detection result", use_container_width=True)
            labels = [model.names[int(b.cls)] for b in results[0].boxes]
            if "shark" in labels:
                output_slot.success("Shark detected in media")
                st.toast('The model detected a Shark', icon="🦈")
            else:
                output_slot.error("No shark detected.")

    # Handle live webcam

    # for "shark" toast message"
    if "shark_alerted" not in st.session_state:
        st.session_state.shark_alerted = False

    if st.session_state.get("webcam_on", False):
        model_live.to("cuda" if torch.cuda.is_available() else "cpu")
        for res in model_live.predict(
            source=0, stream=True,
            conf=conf_threshold, vid_stride=2, iou=0.5
        ):
            if not st.session_state.webcam_on:
                break

            frame = res.plot()
            frame_slot.image(frame, channels="BGR", use_container_width=True)

            # check for shark boxes
            classes = [int(b.cls) for b in res.boxes]
            names   = [model.names[c] for c in classes]
            if "shark" in names and not st.session_state.shark_alerted:
                # first time only:
                st.toast("The model detected a Shark", icon="🦈")
                st.session_state.shark_alerted = True
