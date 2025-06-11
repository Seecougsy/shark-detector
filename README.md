![Project Banner YOLOv8 Shark Detecter](Screenshots/shark-detector-02.png)

This app uses a YOLOv8s model trained on aerial images to detect sharks in real-time video and static photos, showing bounding boxes and confidence scores.

![Shark Detector Application Interface](Screenshots/detector-app2.png)
*Shark Detector Application Interface.*

---

## Project Overview

* **Team Project:** Developed as a university capstone project at Torrens University.
* **Goal:** Built as a real-time marine threat detection tool deployable via drones, aiming to assist lifeguards and emergency response teams with early shark detection.

## Key Features

* **Real-time Detection:** Analyzes live video streams for immediate threat identification.
* **Image Analysis:** Capable of processing individual photos.
* **Adjustable Confidence Threshold:** Allows users to control the sensitivity of detections.
* **Visual Bounding Boxes:** Displays detected sharks with clear bounding boxes and confidence scores.
* **Drone Deployable:** Designed with potential for integration into drone operations.


## Technologies Used

* **Python 3.9+:** The primary programming language.
* **Streamlit:** Used for building the interactive web demonstration application.
* **Ultralytics (YOLOv8s):** The state-of-the-art object detection model for shark identification.
* **OpenCV:** Utilized for image and video processing tasks.
* **Roboflow:** Employed for dataset management, labeling, and robust data augmentation.
* **Google Colab (A100 GPU):** Used for high-performance model training.


## Background
Developed as part of a university capstone project, this application demonstrates the use of AI and computer vision for public safety. `YOLOv8` was selected for its one-pass architecture, which processes entire images in real time, suitable for the  project requirements. It supports early shark detection from aerial drone footage with the goal of assisting lifeguards and emergency response teams.


## The Dataset
We used a custom dataset combining open-source aerial footage and custom footage. Negative examples (e.g., images of surfers, dolphins, and empty ocean scenes) were included to help the model distinguish sharks from other marine objects and reduce false positives. The dataset size was 15907 total images.
Data was split into 70% training, 20% validation, 10% testing.
![Image Augmentation used](Screenshots/data_augmentation.png)


## Training the Model
The notebook [Model_training_notebook.ipynb](Model_training/Model_training_notebook.ipynb) contains the full training pipeline used before integrating the model weights into the application. The model was trained using:

* YOLOv8s from Ultralytics
* Google Colab + A100 GPU
* Roboflow for dataset management and augmentation
* 60 epochs with batch size 64
* Input resolution upscaled to 960×960 for final models

During training, we monitored key metrics to ensure optimal performance:
![Training Results Curves](Screenshots/results.png)
*The training and validation loss curves show the model converging, while precision, recall, and mAP metrics improved steadily over 15 epochs, indicating effective learning and generalization.*

## The Dataset
We used a custom dataset combining open-source aerial footage and custom footage. Negative examples (e.g., images of surfers, dolphins, and empty ocean scenes) were included to help the model distinguish sharks from other marine objects and reduce false positives. The dataset size was **15,907 total images**.
Data was split into 70% training, 20% validation, 10% testing.
![Image Augmentation used](Screenshots/data_augmentation.png)

## Model Performance
????? come back ????? To comprehensively evaluate the model's accuracy and identify areas for improvement, various metrics and visualisations were analyzed on the test set.

**Confusion Matrix:**
![Model Confusion Matrix for Single Class Shark](Screenshots/confusion_matrix.png)
*On the validation set, the model correctly identified **928** shark images (true positives). It also misclassified **148** background images as sharks (false positives) and missed **107** sharks by labelling them as background (false negatives). 









### Key metrics (best run):
`mAP@50: 96.4%`

`Precision: 94.0%`

`Recall: 91.6% `

### Confusion Matrix:



## Running the App
Open Command Prompt (CMD) or Terminal.

1.  **Navigate to the app’s folder.**
    Use the `cd` command. For example:
    ```bash
    cd "/path/to/your/app/folder"
    ```
    (Remember to replace `/path/to/your/app/folder` with your actual directory.)

2.  **Run the app using Streamlit.**
    Execute:
    ```bash
    python -m streamlit run shark.py
    ```
    Your default browser should automatically open the app. If not, copy the URL shown in CMD/Terminal.

3.  **Configure Confidence Level**
    Adjust the "Confidence Level" slider. Only predictions with a confidence score *above* this threshold will be displayed. Higher confidence means fewer, but generally more reliable, detections.

4.  **Upload Content for Detection**
    * **For Image Detection:** Drag and drop or browse to select a `.jpg`, `.jpeg`, or `.png` image. The app will then process and display detections.
    * **For Video Detection:** Drag and drop or browse to select a `.mp4` or `.mpeg4` video. The app will process and display detections in real-time as it plays.

5. **Voilà!**
The model will **detect** if there is a **shark** in the image or video and surround it with **a bounding box** displaying the confidence score.

    ![Shark Detected Example](Screenshots/shark-found.png)
    *A detected shark with its bounding box and confidence score.*



### Requirements
- Python 3.9+
- Streamlit
- Ultralytics
- OpenCV
- Roboflow
## Notes
- This is a proof-of-concept and not production ready
- Intended for educational demonstration and not operational
- All data used to train the model was publicly sourced/openly licensed.