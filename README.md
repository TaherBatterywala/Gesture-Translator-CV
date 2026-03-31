# 🖐️ Gesture Translator — CV Feature Engineering

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

**A lightweight real-time gesture recognition system that ditches deep learning for elegant classical machine learning — powered by geometric feature engineering.**

</div>

---

## 📖 Overview

Most gesture recognition tutorials throw a ResNet or YOLO at the problem and call it a day. This project takes a smarter, leaner approach: instead of learning from raw pixels, we let **MediaPipe** extract 21 precise 3D hand landmarks per frame, then engineer **17 handcrafted geometric features** — distances and joint angles — and feed those compact numbers directly into a **Random Forest Classifier**.

The result is a system that:
- ✅ Runs in **real time** on a standard CPU (no GPU required)
- ✅ Is fully **explainable** — you can inspect every feature's contribution
- ✅ Is **scale-invariant** — works regardless of how close your hand is to the camera
- ✅ Can be **retrained in seconds** with your own custom gestures
- ✅ Has a clean, **modular pipeline** from data collection → training → inference

---

## 🏗️ Architecture

The pipeline has three distinct stages:

```
📷 Webcam Frame
      │
      ▼
┌─────────────────────────────┐
│   MediaPipe Hand Landmarker │  ← Detects 21 (x, y, z) landmarks
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Geometric Feature Engine  │  ← Computes 17 scale-invariant features
│  • 7  Euclidean distances   │
│  • 5  wrist distances       │
│  • 5  PIP joint angles      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Random Forest Classifier   │  ← Predicts gesture label
│  (100 estimators, n=17)     │
└────────────┬────────────────┘
             │
             ▼
    🖥️ On-Screen Prediction
```

### Why No Deep Learning?

| Approach | Model Size | Inference Speed | Explainability | Training Data Needed |
|---|---|---|---|---|
| CNN / ResNet | ~100 MB+ | GPU-dependent | ❌ Black box | Thousands of images |
| **This project** | **~1 MB** | **CPU real-time** | **✅ Full** | **~100–200 frames** |

The geometric features capture the *shape* of the hand pose, exactly what humans use to distinguish gestures.

---

## ✨ The 17 Engineered Features

All distances are **normalized** by the wrist-to-middle-MCP reference distance, making them scale-invariant:

| # | Feature | Landmarks Used | Description |
|---|---|---|---|
| 1 | `thumb_index_dist` | 4 → 8 | Thumb tip to Index tip |
| 2 | `thumb_middle_dist` | 4 → 12 | Thumb tip to Middle tip |
| 3 | `thumb_ring_dist` | 4 → 16 | Thumb tip to Ring tip |
| 4 | `thumb_pinky_dist` | 4 → 20 | Thumb tip to Pinky tip |
| 5 | `index_middle_dist` | 8 → 12 | Adjacent fingertip spread |
| 6 | `middle_ring_dist` | 12 → 16 | Adjacent fingertip spread |
| 7 | `ring_pinky_dist` | 16 → 20 | Adjacent fingertip spread |
| 8 | `thumb_wrist_dist` | 4 → 0 | Thumb extension from wrist |
| 9 | `index_wrist_dist` | 8 → 0 | Index extension from wrist |
| 10 | `middle_wrist_dist` | 12 → 0 | Middle extension from wrist |
| 11 | `ring_wrist_dist` | 16 → 0 | Ring extension from wrist |
| 12 | `pinky_wrist_dist` | 20 → 0 | Pinky extension from wrist |
| 13 | `thumb_angle` | 1, 2, 3 | Thumb IP joint bend angle |
| 14 | `index_angle` | 5, 6, 7 | Index PIP joint bend angle |
| 15 | `middle_angle` | 9, 10, 11 | Middle PIP joint bend angle |
| 16 | `ring_angle` | 13, 14, 15 | Ring PIP joint bend angle |
| 17 | `pinky_angle` | 17, 18, 19 | Pinky PIP joint bend angle |

> **Scale normalization:** All distance features are divided by `dist(landmark[0], landmark[9])` — the wrist-to-middle-MCP distance — so the model works regardless of hand size or camera proximity.

---

## 🖐️ Supported Gestures (Out of the Box)

| Key | Gesture | Description |
|---|---|---|
| `1` | 👍 Thumbs Up | Thumb extended upward, fingers curled |
| `2` | 👎 Thumbs Down | Thumb extended downward, fingers curled |
| `3` | 🖐️ Stop | All five fingers fully extended |
| `4` | ✌️ Peace | Index and middle extended, others curled |
| `5` | ✊ Fist | All fingers tightly curled |

> You can easily add your own gestures — see [Adding Custom Gestures](#adding-custom-gestures).

---

## 📁 Repository Structure

```text
gesture-translator/
├── data/
│   └── processed/
│       └── dataset.csv        # Engineered tabular features (auto-created)
├── models/
│   ├── hand_landmarker.task   # MediaPipe pre-trained hand landmark model
│   └── gesture_model.pkl      # Your trained Random Forest (after training)
├── src/
│   ├── app.py                 # 🎥 Real-time webcam inference
│   ├── collect_data.py        # 📊 Interactive data collection tool
│   ├── feature_extraction.py  # 📐 Geometric feature math (distances & angles)
│   └── train_model.py         # 🤖 Sklearn training & evaluation pipeline
├── tests/
│   └── test_features.py       # ✅ Unit tests for feature extraction math
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- A working webcam
- (Optional) A virtual environment tool

### 1. Clone the Repository

```bash
git clone https://github.com/TaherBatterywala/gesture-translator.git
cd gesture-translator
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**

| Package | Version | Purpose |
|---|---|---|
| `mediapipe` | ≥ 0.10.0 | Hand landmark detection |
| `opencv-python` | ≥ 4.8.0 | Webcam capture & frame display |
| `scikit-learn` | ≥ 1.3.0 | Random Forest classifier |
| `pandas` | ≥ 2.0.0 | Dataset management (CSV I/O) |
| `numpy` | ≥ 1.24.0 | Numerical computations |
| `pytest` | ≥ 7.0.0 | Unit testing |
| `flake8` | ≥ 6.0.0 | Code linting |
| `black` | ≥ 23.0.0 | Code formatting |

---

## 🚀 Usage — Step by Step

### Step 1 · Collect Training Data

Run the interactive data collection tool. A webcam window will open with live hand landmark visualization.

```bash
python src/collect_data.py
```

**Controls:**

| Key | Action |
|---|---|
| `1` | Save current frame as **Thumbs Up** |
| `2` | Save current frame as **Thumbs Down** |
| `3` | Save current frame as **Stop** |
| `4` | Save current frame as **Peace** |
| `5` | Save current frame as **Fist** |
| `Q` | Quit collection |

> 💡 **Tip:** Collect **100–200 samples per gesture** at different hand angles, lighting conditions, and distances for a robust model. The on-screen counter tracks progress per class.

The features are saved incrementally to `data/processed/dataset.csv`.

---

### Step 2 · Train the Model

Train the Random Forest classifier on your collected dataset:

```bash
python src/train_model.py
```

**Example output:**

```
Loading data from: ../data/processed/dataset.csv
Dataset Details: 750 samples with 17 features each.
Unique classes: ['Thumbs Up' 'Thumbs Down' 'Stop' 'Peace' 'Fist']
Training Random Forest Classifier...

Model Accuracy on Test Set: 0.9787

Classification Report:
              precision    recall  f1-score   support
   Thumbs Up       0.98      1.00      0.99        49
Thumbs Down        0.97      0.97      0.97        30
        Stop       1.00      0.97      0.98        30
       Peace       0.97      1.00      0.98        32
        Fist       0.97      0.97      0.97        10

Top 5 Important Features:
 - index_wrist_dist: 0.1842
 - middle_wrist_dist: 0.1731
 - thumb_index_dist: 0.1423
 - ring_wrist_dist: 0.1102
 - pinky_wrist_dist: 0.0987

Saving model to ../models/gesture_model.pkl...
Training Complete!
```

> The **Feature Importances** output is particularly valuable for understanding *which geometric properties* distinguish each gesture — this is a core advantage over black-box deep learning.

---

### Step 3 · Real-Time Inference

Launch the live gesture translator:

```bash
python src/app.py
```

- The webcam opens and landmark skeleton is overlaid on your hand in real-time.
- The predicted gesture label is displayed in the top-left corner.
- Press `Q` to quit.

---

## ➕ Adding Custom Gestures

The system is designed to be easily extensible:

1. **Add your gesture key mapping** in `src/collect_data.py`:
   ```python
   gestures = {
       ord("1"): "Thumbs Up",
       ord("6"): "My New Gesture",  # ← Add here
       ...
   }
   ```

2. **Collect data** by running `collect_data.py` and pressing the new key.

3. **Retrain** with `train_model.py` — the new class is picked up automatically.

No architecture changes, no rewriting layers — just data and a 5-second retrain.

---

## 🧪 Running Tests

Unit tests validate the core geometric math (distance and angle calculations):

```bash
pytest tests/ -v
```

**Tests covered:**

| Test | What it validates |
|---|---|
| `test_calculate_distance` | Euclidean distance: `dist((0,0,0), (3,4,0)) == 5.0` |
| `test_calculate_angle` | 90° and 180° angle computation accuracy |
| `test_extract_features_length` | Output vector is exactly 17 features |
| `test_extract_features_invalid_input` | Raises `ValueError` on wrong landmark count |

---

## 🔍 How Feature Extraction Works

```python
# src/feature_extraction.py — simplified

def extract_features(landmarks):
    # landmarks: list of 21 (x, y, z) tuples from MediaPipe

    # Scale reference: wrist → middle finger MCP
    ref_dist = distance(landmarks[0], landmarks[9])

    features = []

    # 7 inter-tip distances (normalized)
    features += [distance(tip_a, tip_b) / ref_dist for ...]

    # 5 tip-to-wrist distances (normalized)
    features += [distance(tip, landmarks[0]) / ref_dist for tip in tips]

    # 5 PIP joint bending angles (degrees)
    features += [angle(p1, p2_vertex, p3) for each finger]

    return features  # length = 17
```

The **normalization by `ref_dist`** is the key insight — it makes the features invariant to the absolute scale of the hand in the frame, so the model works whether your hand is close to or far from the camera.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Hand Detection | MediaPipe Tasks API (`hand_landmarker.task`) |
| Feature Engineering | NumPy |
| Classifier | scikit-learn `RandomForestClassifier` |
| Webcam & Display | OpenCV |
| Data Storage | Pandas / CSV |
| Testing | pytest |
| Linting / Formatting | flake8, black |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

Please ensure all tests pass (`pytest tests/ -v`) and code is formatted (`black src/ tests/`) before submitting.

---

## 👤 Author

**Taher Batterywala**

> Built as a Computer Vision project demonstrating that elegant feature engineering combined with classical ML can outperform brute-force deep learning approaches in constrained, well-defined domains.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

<div align="center">

*If you found this project helpful or interesting, consider giving it a ⭐ on GitHub!*

</div>
