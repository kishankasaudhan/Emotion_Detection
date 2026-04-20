# Real-Time Emotion Detection

This project uses Python, OpenCV, and a TensorFlow/Keras emotion recognition model to detect faces in a webcam feed and classify emotions in real time.

## Features
- Webcam access using OpenCV
- Real-time face detection using Haar Cascades
- Emotion prediction using a pre-trained CNN model
- Bounding boxes and emotion labels displayed on the video feed
- Supports multiple faces and confidence scores

## Requirements
- Python 3.8+
- OpenCV
- TensorFlow
- NumPy

## Installation
1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install required Python libraries:

```powershell
pip install -r requirements.txt
```

3. Add a pre-trained Keras emotion model named `emotion_model.h5` to the project root.

### Example model sources
- Public FER2013-based Keras models are available from many open-source repositories.
- If you want to train your own model, use FER2013 data and save the model as `emotion_model.h5`.

## Run the project

```powershell
python emotion_detection.py
```

Then point your webcam at one or more faces. The detected emotion label and confidence score will appear over each face.

## Notes
- The code expects a model compatible with 48x48 grayscale face inputs.
- If the model uses 7 classes, emotion labels are: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.
- Press `q` or `ESC` to exit the window.

## Optional improvements
- Replace Haar Cascade face detection with OpenCV DNN for higher accuracy.
- Use a more advanced pre-trained model or transfer learning for improved emotion recognition.
- Add a GUI using Tkinter or Streamlit for nicer controls.
