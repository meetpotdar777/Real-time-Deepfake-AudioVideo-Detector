import os
import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
import threading
from tensorflow.keras.models import load_model

# 1. Suppress TensorFlow logs for a cleaner console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Load Models
# Replace with your actual .h5 file paths
video_model = load_model('deepfake_video_detector.h5')
audio_model = load_model('deepfake_audio_detector.h5')

# Global variables for cross-thread communication
audio_label = "Initialising..."
audio_score = 0.0

def audio_monitor_thread():
    """Captures audio and converts it to a 128x109 Mel Spectrogram."""
    global audio_label, audio_score
    fs = 16000  # Sampling rate
    duration = 1.0  # Capture 1 second of audio
    
    while True:
        try:
            # Record audio
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            audio_data = recording.flatten()
            
            # Generate Mel Spectrogram
            # n_mels=128 matches your model's first dimension
            S = librosa.feature.melspectrogram(y=audio_data, sr=fs, n_mels=128, hop_length=148) 
            # Note: hop_length ~148 at 16k SR roughly results in ~109 time steps for 1s
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Precise Resizing to (128, 109) to prevent ValueErrors
            if S_dB.shape[1] < 109:
                S_dB = np.pad(S_dB, ((0, 0), (0, 109 - S_dB.shape[1])), mode='constant')
            else:
                S_dB = S_dB[:, :109]

            # Reshape for Model: (Batch, Height, Width, Channels)
            input_data = S_dB.reshape(1, 128, 109, 1)
            
            # Prediction
            prediction = audio_model.predict(input_data, verbose=0)
            audio_score = float(prediction[0][0])
            audio_label = "FAKE VOICE" if audio_score > 0.5 else "REAL VOICE"
            
        except Exception as e:
            print(f"Audio Thread Error: {e}")

# Start the audio monitoring in the background
threading.Thread(target=audio_monitor_thread, daemon=True).start()

# --- Main Video Stream ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Process Video Frame
    resized_frame = cv2.resize(frame, (224, 224))
    v_input = np.expand_dims(resized_frame, axis=0) / 255.0
    v_score = video_model.predict(v_input, verbose=0)[0][0]
    v_label = "FAKE FACE" if v_score > 0.5 else "REAL FACE"

    # UI Styling
    color_v = (0, 0, 255) if v_score > 0.5 else (0, 255, 0)
    color_a = (0, 0, 255) if audio_score > 0.5 else (0, 255, 0)
    
    # Dashboard Display
    cv2.rectangle(frame, (10, 10), (420, 120), (0,0,0), -1) # Dark background for text
    cv2.putText(frame, f"VIDEO: {v_label} ({v_score:.2f})", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_v, 2)
    cv2.putText(frame, f"AUDIO: {audio_label} ({audio_score:.2f})", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_a, 2)
    
    # Combined Threat Detection
    if v_score > 0.7 and audio_score > 0.7:
        cv2.putText(frame, "!!! DEEPFAKE ALERT !!!", (150, 450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('2026 Deepfake Security Suite', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()