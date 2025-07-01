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

import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# webstream client

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



class SharkTransformer(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1) Turn the incoming frame into a NumPy array
        img = frame.to_ndarray(format="bgr24")
        # 2) Run YOLO on it
        results = model_live.predict(source=img,
                                     conf=conf_threshold,
                                     iou=0.5)
        # 3) Draw boxes & labels
        annotated = results[0].plot()
        # 4) Convert back to an AV frame
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")



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



# st.markdown(
#     """ <div class="section-box">
#             <div class="section-title">1. Set Detection Confidence Threshold</div>
#             <div class="section-subtitle">Only predictions above this confidence level will be shown.</div>
#             </div>""",
#     unsafe_allow_html=True,
# )

st.subheader("2. Upload or Stream Media")
    # Controls in columns
col1, col3 = st.tabs([":material/upload: Upload File", ":material/camera_video: Live Stream",])

with col1:
    st.markdown("**Upload an image or video** and see YOLO’s shark detections:")
    uploaded = st.file_uploader("Choose media", type=["jpg","png","mp4"])
    upload_slot = st.empty()
    
    if uploaded:
        st.session_state.shark_alerted = False
        name = uploaded.name.lower()

        if name.endswith(".mp4"):
            # save temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(uploaded.read())
            tmp.close()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_live.to(device)

            for res in model_live.predict(
                source=tmp.name,
                stream=True,
                conf=conf_threshold,
                vid_stride=4,
                iou=0.5,
            ):
                frame = res.orig_img.copy()
                for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                (int(x2), int(y2)), (0, 255, 0), 2)
                upload_slot.image(frame, channels="BGR", use_container_width=True)

                names = [model.names[int(b.cls)] for b in res.boxes]
                if "shark" in names and not st.session_state.shark_alerted:
                    st.toast("Shark detected 🦈")
                    st.session_state.shark_alerted = True

            os.remove(tmp.name)

        else:
            # single image
            img = Image.open(uploaded)
            results = model.predict(img,
                                    conf=conf_threshold,
                                    iou=0.5)
            ann = results[0].plot()
            ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            upload_slot.image(ann_rgb,
                            caption="Detection result",
                            use_container_width=True)

            labels = [model.names[int(b.cls)] for b in results[0].boxes]
            if "shark" in labels:
                st.toast("Shark detected")
            else:
                st.error("No shark detected.")

with col3:
    st.markdown("**Live webcam feed** with real-time YOLO")
    st.info("")
    webrtc_streamer(
        key="shark-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=SharkTransformer,
        async_processing=True,
    )