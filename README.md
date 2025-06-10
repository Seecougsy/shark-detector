# Shark Detector - Real-time shark detection
This app uses a YOLOv8s model trained on aerial images to detect sharks in real-time video and static photos, showing bounding boxes and confidence scores.

## Project Overview

- Team Project @ Torrens University
- Built as a real-time marine threat detection tool deployable via drones
- Dataset consisted of aerial marine footage featuring sharks, swimmers, surfers, dolphins, and empty ocean scenes
- Cleaned and labelled images using Roboflow, including manual annotation
- Applied data augmentation (flip, shear, crop, etc.) to reduce overfitting and improve generalisation
- Trained YOLOv8s in Google Colab using a high-performance A100 GPU
- Model performance tracked via mAP, precision, and recall – reaching up to 0.96 mAP@0.5
- Exported the best weights as best.pt and integrated into a Streamlit demo application

## Background
Developed as part of a university capstone project, this application demonstrates the use of AI and computer vision for public safety. It supports early shark detection from aerial drone footage with the goal of assisting lifeguards and emergency response teams.



## The Dataset
We used a custom dataset combining open-source aerial footage and custom footage. Noise images (e.g., surfers, dolphins, background) were added to reduce false positives.
Data was split into 70% training, 20% validation, 10% testing.
![Image Augmentation used](Screenshots/data_augmentation.png)



## Training the Model
The notebook [Model_training_notebook.ipynb](Model_training/Model_training_notebook.ipynb) contains the full training pipeline used before integrating the model weights into the application. The model was trained using: 

- YOLOv8s from Ultralytics
- Google Colab + A100 GPU
- Roboflow for dataset management and augmentation
- 60 epochs with batch size 64
- Input resolution upscaled to 960×960 for final models

**Key metrics (best run):**

- mAP@0.5: 0.960
- Precision: 0.955
- Recall: 0.909
 

## Running the App

Open Command Prompt (CMD) or Terminal.

1. Navigate to the app’s folder. For example:
cd "/Users/calebcougle/Shark_detector"

2. Run the app using Streamlit:
python -m streamlit run shark.py

Your default browser should automatically open the app. If not, copy the URL shown in CMD/Terminal and paste it into your browser.

### Requirements
- Python 3.9+
- Streamlit
- Ultralytics
- OpenCV
- Roboflow
## Notes
- This is a proof-of-conceopt and not production ready
- Intended for educational demonstration and not operational
- All data used to trained the model was publically sourced/ open licence.