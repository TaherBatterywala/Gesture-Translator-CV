import cv2
import mediapipe as mp
import pandas as pd
import os
import sys

# Add src to path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from feature_extraction import extract_features

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def draw_landmarks_on_image(rgb_image, hand_landmarks):
    img = rgb_image.copy()
    h, w, c = img.shape
    px_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for connection in HAND_CONNECTIONS:
        pt1, pt2 = px_landmarks[connection[0]], px_landmarks[connection[1]]
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    for pt in px_landmarks:
        cv2.circle(img, pt, 5, (0, 0, 255), -1)
    return img


def main():
    # Define gesture classes mapping
    gestures = {
        ord("1"): "Thumbs Up",
        ord("2"): "Thumbs Down",
        ord("3"): "Stop",
        ord("4"): "Peace",
        ord("5"): "Fist",
    }

    # Prepare CSV file
    data_file = os.path.join(os.path.dirname(__file__), "../data/processed/dataset.csv")
    columns = [
        "thumb_index_dist",
        "thumb_middle_dist",
        "thumb_ring_dist",
        "thumb_pinky_dist",
        "index_middle_dist",
        "middle_ring_dist",
        "ring_pinky_dist",
        "thumb_wrist_dist",
        "index_wrist_dist",
        "middle_wrist_dist",
        "ring_wrist_dist",
        "pinky_wrist_dist",
        "thumb_angle",
        "index_angle",
        "middle_angle",
        "ring_angle",
        "pinky_angle",
        "label",
    ]

    if not os.path.exists(data_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(data_file, index=False)

    # Setup MediaPipe Tasks HandLandmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = os.path.join(
        os.path.dirname(__file__), "../models/hand_landmarker.task"
    )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    print("Webcam Object Created. Press appropriate key to save gesture:")
    for key, name in gestures.items():
        print(f"  '{chr(key)}' -> {name}")
    print("Press 'Q' to quit.")

    features_collected = {name: 0 for name in gestures.values()}

    with HandLandmarker.create_from_options(options) as detector:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Use Task API to process frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = detector.detect(mp_image)

            features = None
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    # Draw
                    img = draw_landmarks_on_image(img, hand_landmarks)

                    # Convert to standard format
                    h, w, c = img.shape
                    landmarks = [
                        (lm.x * w, lm.y * h, lm.z * w) for lm in hand_landmarks
                    ]

                    try:
                        features = extract_features(landmarks)
                    except Exception as e:
                        print(f"Error extracting features: {e}")
                        features = None

            cv2.putText(
                img,
                "Data Collection Mode",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_offset = 60
            for name, count in features_collected.items():
                cv2.putText(
                    img,
                    f"{name}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    1,
                )
                y_offset += 25

            cv2.imshow("Hand Tracking Data Collection", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in gestures and features is not None:
                label = gestures[key]
                row = features + [label]
                df_row = pd.DataFrame([row], columns=columns)
                df_row.to_csv(data_file, mode="a", header=False, index=False)
                features_collected[label] += 1
                print(f"Saved {label} - Total: {features_collected[label]}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
