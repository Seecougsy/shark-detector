# Shark Detector - Real-time shark detection
This app uses a YOLOv8s model trained on aerial images to detect sharks in real-time video and static photos, showing bounding boxes and confidence scores.

# Overview
Developed as part of a university group project, this drone-deployable shark detection system uses Python and YOLOv8 to identify marine threats in real time using a model we trained. The intention is to support coastal safety by enabling automated monitoring along Australian shorelines. 

# Model Training
We used publicly available aerial marine datasets and trained a custom YOLOv8 object detection model via Roboflow. The training process involved:

- Label management, data augmentation, and evaluation

- Exported the best model as a .pt file integrated into the Streamlit app

# Launch

Open Command Prompt (CMD) or Terminal.

1. Navigate to the app’s folder. For example:
cd "/Users/calebcougle/Shark_detector"

2. Run the app using Streamlit:
python -m streamlit run shark.py

Your default browser should automatically open the app. If not, copy the URL shown in CMD/Terminal and paste it into your browser.
