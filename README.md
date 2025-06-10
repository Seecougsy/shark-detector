# Shark Detector - Real-time shark detection
This app uses a YOLOv8s model trained on aerial images to detect sharks in real-time video and static photos, showing bounding boxes and confidence scores.
![Shark Detector App](Screenshots/detector-app2.png)
*Shark Detector Application.*

![Shark Detector App](Screenshots/shark-found.png)
*The Output.*




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
We used a custom dataset combining open-source aerial footage and custom footage. Negative examples (e.g., images of surfers, dolphins, and empty ocean scenes) were included to help the model distinguish sharks from other marine objects and reduce false positives.
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