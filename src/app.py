import cv2
import mediapipe as mp
import pickle
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
    model_path = os.path.join(os.path.dirname(__file__), "../models/gesture_model.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    with open(model_path, "rb") as f:
        rf_clf = pickle.load(f)

    print("Model Loaded. Starting Webcam...")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    mp_model_path = os.path.join(
        os.path.dirname(__file__), "../models/hand_landmarker.task"
    )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mp_model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    with HandLandmarker.create_from_options(options) as detector:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process using Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = detector.detect(mp_image)
            gesture_prediction = "None"

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    img = draw_landmarks_on_image(img, hand_landmarks)
                    h, w, c = img.shape
                    landmarks = [
                        (lm.x * w, lm.y * h, lm.z * w) for lm in hand_landmarks
                    ]

                    try:
                        features = extract_features(landmarks)
                        prediction = rf_clf.predict([features])
                        gesture_prediction = prediction[0]
                    except Exception as e:
                        gesture_prediction = f"Error: {str(e)}"

            cv2.putText(
                img,
                f"Prediction: {gesture_prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Gesture Translator", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
