import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTION_MODEL_PATH = "emotion_model.h5"
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_face_detector():
    if not os.path.exists(FACE_CASCADE_PATH):
        raise FileNotFoundError(f"Haar cascade XML not found: {FACE_CASCADE_PATH}")
    return cv2.CascadeClassifier(FACE_CASCADE_PATH)


def load_emotion_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please download or create a compatible Keras model and place it in this folder.")
        print("Example filename: emotion_model.h5")
        sys.exit(1)
    return load_model(model_path, compile=False)


def preprocess_face(face_gray):
    face_resized = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
    face_normalized = face_resized.astype("float32") / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=-1)
    face_batch = np.expand_dims(face_expanded, axis=0)
    return face_batch


def annotate_frame(frame, x, y, w, h, label_text, confidence):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Label background
    label = f"{label_text}: {confidence:.2f}"
    label_bg_color = (0, 255, 0)
    label_text_color = (0, 0, 0)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    rectangle_btm_right = (x + text_size[0] + 10, y - 10 if y - 10 > 20 else y + h + 25)
    rectangle_top_left = (x, y - 25 if y - 25 > 0 else y + h + 5)
    cv2.rectangle(frame, rectangle_top_left, rectangle_btm_right, label_bg_color, cv2.FILLED)
    cv2.putText(
        frame,
        label,
        (rectangle_top_left[0] + 5, rectangle_top_left[1] + text_size[1] + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        label_text_color,
        1,
        cv2.LINE_AA,
    )


def main():
    face_detector = load_face_detector()
    emotion_model = load_emotion_model(EMOTION_MODEL_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting real-time emotion detection. Press 'q' or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (x, y, w, h) in faces:
            face_gray = gray[y : y + h, x : x + w]
            face_input = preprocess_face(face_gray)
            predictions = emotion_model.predict(face_input, verbose=0)[0]
            max_index = int(np.argmax(predictions))
            confidence = float(predictions[max_index])
            emotion_label = EMOTION_LABELS[max_index]
            annotate_frame(frame, x, y, w, h, emotion_label, confidence)

        cv2.imshow("Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
