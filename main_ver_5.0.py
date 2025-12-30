import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
import threading
from collections import deque
from tensorflow.keras.models import load_model

# --- 1. SETTINGS & SMOOTHING ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
V_THRESHOLD = 0.88  # High threshold for high accuracy
A_THRESHOLD = 0.85 

# Buffers prevent the "flickering" you saw earlier
v_buffer = deque(maxlen=20)
a_buffer = deque(maxlen=20)

try:
    v_model = load_model('deepfake_video_detector.h5')
    a_model = load_model('deepfake_audio_detector.h5')
    print("✅ ENGINES ONLINE: Spatio-Temporal Analysis Active.")
except Exception as e:
    print(f"❌ Model Error: {e}")
    exit()

# MediaPipe is much better than Haar for distinguishing humans from cartoons
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)

audio_stats = {"score": 0.0, "label": "CALIBRATING"}

# --- 2. ADAPTIVE AUDIO PROCESSING ---
def audio_engine():
    global audio_stats
    fs, duration = 16000, 1.2
    
    # Adaptive noise floor
    print("🎙️ Calibrating Microphone...")
    while True:
        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            audio_data = recording.flatten()
            
            # BIOMETRIC GATE: Ignore background hum/static
            # Real speech has a much higher energy peak than background static
            if np.max(np.abs(audio_data)) < 0.04: 
                a_buffer.append(0.0) 
                audio_stats["label"] = "SILENCE/REAL"
                continue

            # Advanced Pre-processing
            audio_data = librosa.util.normalize(audio_data)
            spec = librosa.feature.melspectrogram(y=audio_data, sr=fs, n_mels=128, hop_length=176)
            spec_db = librosa.power_to_db(spec, ref=np.max)
            
            # Match model dimensions (128, 109)
            if spec_db.shape[1] < 109:
                spec_db = np.pad(spec_db, ((0, 0), (0, 109 - spec_db.shape[1])), mode='constant')
            else:
                spec_db = spec_db[:, :109]

            # Predict and Smooth
            raw_a = a_model.predict(spec_db.reshape(1, 128, 109, 1), verbose=0)[0][0]
            a_buffer.append(raw_a)
            
            audio_stats["score"] = sum(a_buffer) / len(a_buffer)
            audio_stats["label"] = "FAKE AUDIO" if audio_stats["score"] > A_THRESHOLD else "REAL AUDIO"
            
        except Exception: pass

threading.Thread(target=audio_engine, daemon=True).start()

# --- 3. BIOMETRIC VIDEO PROCESSING ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Human Landmark Check
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    
    v_final_score = 0.0
    v_final_label = "NO HUMAN"

    if results.detections:
        for det in results.detections:
            # Crop Face
            bbox = det.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            
            face_img = frame[max(0, y):y+h, max(0, x):x+w]
            if face_img.size > 0:
                # 2. Texture & Pixel Analysis
                face_input = cv2.resize(face_img, (224, 224)) / 255.0
                raw_v = v_model.predict(np.expand_dims(face_input, axis=0), verbose=0)[0][0]
                v_buffer.append(raw_v)
                
                v_final_score = sum(v_buffer) / len(v_buffer)
                v_final_label = "DEEPFAKE" if v_final_score > V_THRESHOLD else "AUTHENTIC"
                
                # High-Tech Display
                color = (0, 0, 255) if v_final_label == "DEEPFAKE" else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, v_final_label, (x, y-10), 1, 1.5, color, 2)

    # --- 4. HUD OVERLAY ---
    # Draw a black bar at the bottom for status
    cv2.rectangle(frame, (0, frame.shape[0]-80), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
    
    a_color = (0, 0, 255) if audio_stats["score"] > A_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, f"MIC: {audio_stats['label']} ({audio_stats['score']:.2f})", (20, frame.shape[0]-45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, a_color, 2)
    
    v_color = (0, 0, 255) if v_final_score > V_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, f"CAM: {v_final_label} ({v_final_score:.2f})", (20, frame.shape[0]-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)

    cv2.imshow('Deepfake Security Suite 2026', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()