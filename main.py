import cv2
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.models import load_model

# Load pre-trained models (Ensure you have trained/downloaded these .h5 files)
video_model = load_model('deepfake_video_detector.h5')
audio_model = load_model('deepfake_audio_detector.h5')

def analyze_frame(frame):
    """Detects visual artifacts in a single frame."""
    # Preprocessing: Resize and normalize
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(resized_frame, axis=0) / 255.0
    
    prediction = video_model.predict(img_array)
    return prediction[0][0]  # Returns probability (0 = Real, 1 = Fake)

def analyze_audio(audio_path):
    """Analyzes audio frequency for synthetic patterns."""
    y, sr = librosa.load(audio_path, sr=16000)
    # Extract MFCC features (standard for audio deepfake detection)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, 40, 1)
    
    prediction = audio_model.predict(mfccs_processed)
    return prediction[0][0]

# --- Real-Time Execution ---
cap = cv2.VideoCapture(0) # Capture from Webcam

while True:
    ret, frame = cap.read()
    if not ret: break

    # Analyze visual consistency
    score = analyze_frame(frame)
    label = "DEEPFAKE" if score > 0.5 else "REAL"
    color = (0, 0, 255) if label == "DEEPFAKE" else (0, 255, 0)

    # Display results on the live feed
    cv2.putText(frame, f"{label} (Score: {score:.2f})", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Real-time Deepfake Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()