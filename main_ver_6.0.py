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

# --- 1. SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
V_THRESHOLD = 0.85
A_THRESHOLD = 0.82

# Buffers for stability
v_buffer = deque(maxlen=20)
a_buffer = deque(maxlen=15)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

try:
    v_model = load_model('deepfake_video_detector.h5')
    a_model = load_model('deepfake_audio_detector.h5')
    print("✅ FORENSIC SUITE ONLINE (MediaPipe Engine)")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

audio_stats = {"score": 0.0, "label": "SCANNING"}

# --- 2. AUDIO FORENSICS ---
def audio_engine():
    global audio_stats
    fs, duration = 16000, 1.2
    while True:
        try:
            rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            audio = rec.flatten()
            
            if np.max(np.abs(audio)) < 0.03: # Noise Gate
                a_buffer.append(0.0)
                continue

            audio = librosa.util.normalize(audio)
            spec = librosa.feature.melspectrogram(y=audio, sr=fs, n_mels=128, hop_length=176)
            spec_db = librosa.power_to_db(spec, ref=np.max)
            
            # Reshape to (1, 128, 109, 1)
            inp = np.pad(spec_db, ((0,0),(0,109-spec_db.shape[1])))[:,:109].reshape(1,128,109,1)
            
            score = a_model.predict(inp, verbose=0)[0][0]
            a_buffer.append(score)
            avg_s = sum(a_buffer)/len(a_buffer)
            audio_stats = {"score": avg_s, "label": "FAKE" if avg_s > A_THRESHOLD else "REAL"}
        except: pass

threading.Thread(target=audio_engine, daemon=True).start()

# --- 3. VIDEO LIVENESS & TEXTURE SCAN ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    v_score_final = 0.0
    status = "NO HUMAN"

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # 1. Geometry Check (Is it a cartoon?)
            # Cartoons lack depth in landmarks. We measure standard deviation of Z-axis.
            z_depth = np.std([lm.z for lm in landmarks.landmark])
            
            # 2. Extract Face ROI
            h, w, _ = frame.shape
            x_pts = [lm.x * w for lm in landmarks.landmark]
            y_pts = [lm.y * h for lm in landmarks.landmark]
            x1, y1, x2, y2 = int(min(x_pts)), int(min(y_pts)), int(max(x_pts)), int(max(y_pts))
            
            face_roi = frame[max(0,y1):y2, max(0,x1):x2]
            if face_roi.size > 0:
                face_input = cv2.resize(face_roi, (224, 224)) / 255.0
                raw_v = v_model.predict(np.expand_dims(face_input, axis=0), verbose=0)[0][0]
                
                # Biometric Logic: Cartoons have very low depth variance (Z-axis)
                if z_depth < 0.02: # Threshold for 'flatness'
                    status = "CARTOON/FLAT IMAGE DETECTED"
                    v_buffer.append(0.0)
                else:
                    v_buffer.append(raw_v)
                    status = "AUTHENTIC" if (sum(v_buffer)/len(v_buffer)) < V_THRESHOLD else "DEEPFAKE"
                
                v_score_final = sum(v_buffer)/len(v_buffer)
                
                color = (0,0,255) if "FAKE" in status or "DEEPFAKE" in status else (0,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, status, (x1, y1-10), 1, 1.2, color, 2)

    # UI Panel
    cv2.rectangle(frame, (10, 10), (450, 120), (0,0,0), -1)
    cv2.putText(frame, f"AUDIO: {audio_stats['label']} ({audio_stats['score']:.2f})", (20, 50), 1, 1.2, (255,255,255), 2)
    cv2.putText(frame, f"VIDEO: {v_score_final:.2f} ({status})", (20, 100), 1, 1.2, (255,255,255), 2)

    cv2.imshow('Exact Deepfake Detector (No-Dlib Version)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()