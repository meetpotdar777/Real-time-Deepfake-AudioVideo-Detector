import os
import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
import threading
from tensorflow.keras.models import load_model

# --- 1. CONFIGURATION & PRE-CHECKS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
THRESHOLD_AUDIO = 0.80  # Increased to 0.80 to reduce False Positives
THRESHOLD_VIDEO = 0.65

try:
    video_model = load_model('deepfake_video_detector.h5')
    audio_model = load_model('deepfake_audio_detector.h5')
    print("✅ System Ready: Models Loaded.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")
    exit()

# Shared state
audio_label = "CALIBRATING..."
audio_score = 0.0

# --- 2. ENHANCED AUDIO THREAD ---
def audio_monitor_thread():
    global audio_label, audio_score
    fs = 16000
    duration = 1.0 
    
    print("🎙️ Audio Monitoring: ACTIVE")
    while True:
        try:
            # 1. Capture Raw Audio
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            audio_data = recording.flatten()
            
            # 2. NOISE GATE: If room is too quiet, skip processing
            if np.max(np.abs(audio_data)) < 0.02:
                audio_label = "SILENT"
                audio_score = 0.0
                continue
            
            # 3. NORMALIZATION: Fixes volume-based false positives
            audio_data = librosa.util.normalize(audio_data)
            
            # 4. SPECTROGRAM (Target Shape: 128 x 109)
            # hop_length=148 at 16k SR creates ~109 time steps
            S = librosa.feature.melspectrogram(y=audio_data, sr=fs, n_mels=128, hop_length=148)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # 5. RESIZE: Ensure perfect 109 width for model input compatibility
            if S_dB.shape[1] < 109:
                S_dB = np.pad(S_dB, ((0, 0), (0, 109 - S_dB.shape[1])), mode='constant')
            else:
                S_dB = S_dB[:, :109]

            # 6. INFERENCE
            input_data = S_dB.reshape(1, 128, 109, 1)
            prediction = audio_model.predict(input_data, verbose=0)
            audio_score = float(prediction[0][0])
            
            # Calibrated Labeling
            audio_label = "FAKE VOICE" if audio_score > THRESHOLD_AUDIO else "REAL VOICE"
            
        except Exception as e:
            print(f"Audio Error: {e}")

threading.Thread(target=audio_monitor_thread, daemon=True).start()

# --- 3. VIDEO PROCESSING & UI ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Video Preprocessing
    small_frame = cv2.resize(frame, (224, 224))
    v_input = np.expand_dims(small_frame, axis=0) / 255.0
    v_score = video_model.predict(v_input, verbose=0)[0][0]
    v_label = "FAKE FACE" if v_score > THRESHOLD_VIDEO else "REAL FACE"

    # UI Feedback
    c_v = (0, 0, 255) if v_score > THRESHOLD_VIDEO else (0, 255, 0)
    c_a = (0, 0, 255) if audio_score > THRESHOLD_AUDIO else (0, 255, 0)
    
    # Dashboard Background
    cv2.rectangle(frame, (10, 10), (460, 140), (20, 20, 20), -1)
    
    # Status Text
    cv2.putText(frame, f"VIDEO SCORE: {v_score:.2f} [{v_label}]", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_v, 2)
    cv2.putText(frame, f"AUDIO SCORE: {audio_score:.2f} [{audio_label}]", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_a, 2)
    
    # Logic for Multimodal Alert
    if v_score > 0.8 and audio_score > 0.8:
        cv2.putText(frame, "!!! SYNTHETIC MEDIA ALERT !!!", (100, 420), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow('Security Shield: AI Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()