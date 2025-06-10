# Shark Detector - Real-time shark detection
This app uses a YOLOv8s model trained on aerial images to detect sharks in real-time video and static photos, showing bounding boxes and confidence scores.

## Project Overview

- Team Project @ Torrens University
- Collected and cleaned a large aerial marine dataset using public sources on found on 

- Applied image augmentation via Roboflow to improve generalisation and reduce overfitting

- Trained a YOLOv8s object detection model and exported the best-performing .pt file

- Integrated the trained model into a demo application using Streamlit

- Intended for deployment on drones to support beach safety

## Background
Developed as part of a university group project, this application demonstrates the practical use of AI and computer vision for public safety. The aim was to build a real-time marine threat detection tool deployable via drones, focusing on Australia's beaches so that.


## Model Training
We used publicly available aerial datasets and trained a custom object detection model using Roboflow and Ultralytics YOLOv8. Training involved:

- Data annotation and image augmentation
- Model training, evaluation, and tuning
- Exporting the final model as `best.pt`


## Running the App

Open Command Prompt (CMD) or Terminal.

1. Navigate to the app’s folder. For example:
cd "/Users/calebcougle/Shark_detector"

2. Run the app using Streamlit:
python -m streamlit run shark.py

Your default browser should automatically open the app. If not, copy the URL shown in CMD/Terminal and paste it into your browser.

### Requirements

## Notes