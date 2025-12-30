import os
import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
import threading
from collections import deque
from tensorflow.keras.models import load_model

# --- 1. SETTINGS & STABILITY BUFFERS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
AUDIO_THRESHOLD = 0.85  # Higher threshold = more strict (less likely to call real voice 'fake')
VIDEO_THRESHOLD = 0.70

# Buffers store the last 10 scores to provide a smooth average
video_buffer = deque(maxlen=10)
audio_buffer = deque(maxlen=10)

try:
    video_model = load_model('deepfake_video_detector.h5')
    audio_model = load_model('deepfake_audio_detector.h5')
    print("✅ Security Core Loaded.")
except Exception as e:
    print(f"❌ Critical Error: {e}")
    exit()

audio_final_label = "READY"
audio_final_score = 0.0

# --- 2. ADVANCED AUDIO MONITOR (With Smoothing) ---
def audio_monitor_thread():
    global audio_final_label, audio_final_score
    fs, duration = 16000, 1.0
    
    while True:
        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            audio_data = recording.flatten()
            
            # NOISE GATE: Only process if there is actual speech (Volume > 0.03)
            if np.max(np.abs(audio_data)) < 0.03:
                audio_buffer.append(0.0)
                continue
            
            # Preprocessing
            audio_data = librosa.util.normalize(audio_data)
            S = librosa.feature.melspectrogram(y=audio_data, sr=fs, n_mels=128, hop_length=148)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Resizing to 128x109
            if S_dB.shape[1] < 109:
                S_dB = np.pad(S_dB, ((0, 0), (0, 109 - S_dB.shape[1])), mode='constant')
            else:
                S_dB = S_dB[:, :109]

            # Prediction
            input_data = S_dB.reshape(1, 128, 109, 1)
            raw_score = audio_model.predict(input_data, verbose=0)[0][0]
            
            # Add to smoothing buffer
            audio_buffer.append(raw_score)
            audio_final_score = sum(audio_buffer) / len(audio_buffer)
            audio_final_label = "FAKE VOICE" if audio_final_score > AUDIO_THRESHOLD else "REAL VOICE"
            
        except Exception: pass

threading.Thread(target=audio_monitor_thread, daemon=True).start()

# --- 3. VIDEO SCANNER (With Face Detection) ---
# Using a Haar Cascade to ensure we ONLY analyze human faces, not cartoons or backgrounds
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    v_label = "NO FACE"
    v_display_score = 0.0

    for (x, y, w, h) in faces:
        # Crop to the face area ONLY
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (224, 224)) / 255.0
        v_input = np.expand_dims(face_resized, axis=0)
        
        raw_v_score = video_model.predict(v_input, verbose=0)[0][0]
        video_buffer.append(raw_v_score)
        
        # Calculate smoothed score
        v_display_score = sum(video_buffer) / len(video_buffer)
        v_label = "FAKE FACE" if v_display_score > VIDEO_THRESHOLD else "REAL FACE"
        
        # Draw box around the detected face
        color = (0, 0, 255) if v_label == "FAKE FACE" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{v_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- 4. DASHBOARD OVERLAY ---
    cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"AUDIO CONF: {audio_final_score:.2f} [{audio_final_label}]", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"VIDEO CONF: {v_display_score:.2f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Calibrated Deepfake Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()