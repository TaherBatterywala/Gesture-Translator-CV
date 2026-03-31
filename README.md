# Sign Language / Gesture Translator 

> **Project — Computer Vision via Feature Engineering**

A lightweight, highly accurate custom gesture translator that uses classical machine learning instead of massive deep learning models. By converting raw video data into structured tabular geometric features (angles and Euclidean distances), this project demonstrates a highly robust edge-friendly pipeline.

---

##  Architecture

Instead of feeding raw pixels into a ResNet or YOLO, we use **MediaPipe** solely for Landmark Extraction. 
We then engineer **17 specific geometric features** per frame:
1. Distance between all specific fingertips and the wrist.
2. Distances between adjacent fingers.
3. Angles of the PIP joints to detect if fingers are folded.

These flat numerical features are then passed into a **Random Forest Classifier**, creating a blazingly fast and explainable AI system.

##  Repository Structure

```text
gesture_translator/
├── .github/workflows/
│   └── ci.yml             # Automated Testing & Linting
├── data/
│   ├── raw/               # (Git-ignored) Where videos would sit
│   └── processed/         # Engineered tabular features (dataset.csv)
├── models/                # Saved trained Random Forest model (gesture_model.pkl)
├── notebooks/             
├── src/
│   ├── app.py                 # Real-time webcam inference
│   ├── collect_data.py        # Custom data gathering tool
│   ├── feature_extraction.py  # Geometric logic
│   └── train_model.py         # Sklearn training pipeline
├── tests/
│   └── test_features.py       # Unit tests for our math functions
├── requirements.txt       # Dependencies
└── README.md
```

##  Setup Instructions

1. **Clone the repository and CD into it**:
```bash
git clone <https://github.com/Swayam-Burde/gesture-translator.git>
cd gesture_translator
```

2. **Create a virtual environment (Optional but Recommended)**:
```bash
python -m venv venv
source venv/bin/activate  
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

### Step 1: Data Collection
Run the data collection script to record your hand doing specific gestures:
```bash
python src/collect_data.py
```
- A webcam window will pop up.
- Show your hand to the camera.
- Press `1`, `2`, `3`, `4`, or `5` to save the current frame's extracted features as a specific gesture (e.g., Thumbs Up, Stop, Peace).
- Press `Q` when you have collected enough data (recommend 100-200 frames per gesture at various angles).

### Step 2: Model Training
Train the Random Forest model on the tabular CSV data you just generated:
```bash
python src/train_model.py
```
This script will output your Model Accuracy, a Classification Report, and **Feature Importances** (showing you mathematically which angles or distances the ML used most!). It saves the result to `models/gesture_model.pkl`.

### Step 3: Real-Time Inference
Run the gesture translator!
```bash
python src/app.py
```
The webcam will open again, run the MediaPipe + math pipeline, predict your gesture using the `.pkl` model, and display the English translation on screen.
