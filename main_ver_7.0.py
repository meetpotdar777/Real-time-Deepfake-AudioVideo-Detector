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

# --- 1. SETUP & CONFIG ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
V_THRESHOLD, A_THRESHOLD = 0.85, 0.82

# Buffers for stability (Temporal Smoothing)
v_buffer = deque(maxlen=20)
a_buffer = deque(maxlen=15)

# Initialize MediaPipe with proper dimensions to fix the "NORM_RECT" warning
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

try:
    # Use compile=False to avoid the "Optimizer state" warnings
    v_model = load_model('deepfake_video_detector.h5', compile=False)
    a_model = load_model('deepfake_audio_detector.h5', compile=False)
    print("✅ FORENSIC SUITE ONLINE")
except Exception as e:
    print(f"❌ Load Error: {e}"); exit()

audio_stats = {"score": 0.0, "label": "CALIBRATING..."}

# --- 2. AUDIO ENGINE (Fixed Negative Index & Mic Error) ---
def audio_engine():
    global audio_stats
    fs, duration = 16000, 1.2
    
    # Auto-find default device
    try:
        device_info = sd.query_devices(kind='input')
        mic_idx = device_info['index']
        print(f"🎙️ Mic Active: {device_info['name']}")
    except:
        print("❌ No Microphone detected!"); return

    while True:
        try:
            # 1. Capture Audio
            rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=mic_idx)
            sd.wait()
            audio = rec.flatten()
            
            # 2. FIXED: Advanced Noise Gate (Prevents negative index errors)
            # If the signal is too low, we don't process it at all
            if np.max(np.abs(audio)) < 0.02:
                audio_stats["label"] = "SILENCE (REAL)"
                a_buffer.append(0.0)
                continue

            # 3. Normalization & Feature Extraction
            audio = librosa.util.normalize(audio)
            
            # Use a higher n_fft for better resolution
            spec = librosa.feature.melspectrogram(y=audio, sr=fs, n_mels=128, hop_length=176, n_fft=2048)
            
            # Fixed: Ensure no zero values before log conversion to prevent negative/inf values
            spec_db = librosa.power_to_db(spec + 1e-9, ref=np.max)
            
            # 4. Shape Alignment (Exact (1, 128, 109, 1))
            if spec_db.shape[1] < 109:
                inp_data = np.pad(spec_db, ((0,0),(0,109-spec_db.shape[1])), mode='constant')
            else:
                inp_data = spec_db[:, :109]
                
            inp_data = inp_data.reshape(1, 128, 109, 1)
            
            # 5. Prediction
            score = a_model.predict(inp_data, verbose=0)[0][0]
            a_buffer.append(score)
            
            avg_s = sum(a_buffer)/len(a_buffer)
            audio_stats["score"] = avg_s
            audio_stats["label"] = "FAKE VOICE" if avg_s > A_THRESHOLD else "REAL VOICE"
            
        except Exception as e:
            # Catching the error properly to prevent thread death
            audio_stats["label"] = "RECONNECTING MIC..."
            print(f"⚠️ Mic System Reset: {e}")

threading.Thread(target=audio_engine, daemon=True).start()

# --- 3. VIDEO ENGINE ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape

    # MediaPipe Processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    v_score_final = 0.0
    v_status = "NO HUMAN"

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # 1. Z-Depth Authenticity (Exactness Layer)
            z_depth = np.std([lm.z for lm in landmarks.landmark])
            
            # 2. Face ROI Extraction
            x_pts = [lm.x * w for lm in landmarks.landmark]
            y_pts = [lm.y * h for lm in landmarks.landmark]
            x1, y1, x2, y2 = int(min(x_pts)), int(min(y_pts)), int(max(x_pts)), int(max(y_pts))
            
            # Add padding to crop
            padding = 20
            face_roi = frame[max(0,y1-padding):min(h,y2+padding), max(0,x1-padding):min(w,x2+padding)]
            
            if face_roi.size > 0:
                face_input = cv2.resize(face_roi, (224, 224)) / 255.0
                raw_v = v_model.predict(np.expand_dims(face_input, axis=0), verbose=0)[0][0]
                
                # 3. Logic: Filter Cartoons (Flat) vs Human (3D)
                if z_depth < 0.012: # Threshold for 2D/Cartoon rejection
                    v_status = "FLAT/CARTOON (REAL)"
                    v_buffer.append(0.0)
                else:
                    v_buffer.append(raw_v)
                    v_status = "DEEPFAKE" if (sum(v_buffer)/len(v_buffer)) > V_THRESHOLD else "AUTHENTIC"
                
                v_score_final = sum(v_buffer)/len(v_buffer)
                
                # Draw UI
                color = (0,0,255) if "DEEPFAKE" in v_status else (0,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, v_status, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    # --- 4. ULTIMATE HUD ---
    cv2.rectangle(frame, (0, 0), (500, 140), (15, 15, 15), -1) # Dark glass effect
    
    # Audio status
    a_color = (0, 0, 255) if audio_stats["score"] > A_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, f"AUDIO: {audio_stats['label']} ({audio_stats['score']:.2f})", 
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, a_color, 2)
    
    # Video status
    v_color = (0, 0, 255) if v_score_final > V_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, f"VIDEO: {v_status} ({v_score_final:.2f})", 
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, v_color, 2)

    cv2.imshow('Exact Deepfake Forensic Detector v8.0', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()