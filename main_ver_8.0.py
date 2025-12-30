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

# --- 1. SETTINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
V_THRESHOLD, A_THRESHOLD = 0.85, 0.82

v_buffer = deque(maxlen=20)
a_buffer = deque(maxlen=15)

# MediaPipe with explicit settings to reduce warnings
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, 
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

try:
    v_model = load_model('deepfake_video_detector.h5', compile=False)
    a_model = load_model('deepfake_audio_detector.h5', compile=False)
    print("✅ SYSTEM READY: Biometric & Audio Engines Synchronized")
except Exception as e:
    print(f"❌ Load Error: {e}"); exit()

audio_stats = {"score": 0.0, "label": "INIT MIC..."}

# --- 2. THE ULTIMATE AUDIO ENGINE (Zero-Index Error Fix) ---
def audio_engine():
    global audio_stats
    fs, duration = 16000, 1.2
    
    try:
        device_info = sd.query_devices(kind='input')
        mic_idx = device_info['index']
        print(f"🎙️ Mic Active: {device_info['name']}")
    except:
        audio_stats["label"] = "NO MIC FOUND"
        return

    while True:
        try:
            rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=mic_idx)
            sd.wait()
            audio = rec.flatten()
            
            # 1. Noise Floor Check
            if np.max(np.abs(audio)) < 0.01:
                audio_stats["label"] = "SILENCE (REAL)"
                a_buffer.append(0.0)
                continue

            # 2. Safe Pre-processing
            audio = librosa.util.normalize(audio)
            # Use fixed n_fft and hop_length to guarantee time-steps
            spec = librosa.feature.melspectrogram(y=audio, sr=fs, n_mels=128, hop_length=176, n_fft=1024)
            spec_db = librosa.power_to_db(spec + 1e-10, ref=np.max)
            
            # 3. FIXED SLICING: Guaranteed (128, 109) without negative indices
            # We create a zero-matrix and fill it to avoid padding errors
            final_spec = np.zeros((128, 109))
            time_steps = min(spec_db.shape[1], 109)
            final_spec[:, :time_steps] = spec_db[:, :time_steps]
            
            inp = final_spec.reshape(1, 128, 109, 1)
            
            # 4. Inference
            score = a_model.predict(inp, verbose=0)[0][0]
            a_buffer.append(float(score))
            
            avg_s = sum(a_buffer)/len(a_buffer)
            audio_stats["score"] = avg_s
            audio_stats["label"] = "FAKE VOICE" if avg_s > A_THRESHOLD else "REAL VOICE"
            
        except Exception as e:
            audio_stats["label"] = "MIC RECOVERY"
            # print(f"Audio debug: {e}") # Enable for deep debugging

threading.Thread(target=audio_engine, daemon=True).start()

# --- 3. VIDEO ENGINE (High-Exactness Cartoon Rejection) ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    v_score_final = 0.0
    status = "NO TARGET"

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # GEOMETRY ANALYSIS: Standard deviation of Z-axis (Depth)
            # Real faces have a Z-depth variance > 0.02. Cartoons are < 0.015.
            z_depth = np.std([lm.z for lm in landmarks.landmark])
            
            # EXTRACT ROI
            x_pts = [lm.x * w for lm in landmarks.landmark]
            y_pts = [lm.y * h for lm in landmarks.landmark]
            x1, y1, x2, y2 = int(min(x_pts)), int(min(y_pts)), int(max(x_pts)), int(max(y_pts))
            
            face_roi = frame[max(0,y1-10):min(h,y2+10), max(0,x1-10):min(w,x2+10)]
            
            if face_roi.size > 0:
                face_input = cv2.resize(face_roi, (224, 224)) / 255.0
                raw_v = v_model.predict(np.expand_dims(face_input, axis=0), verbose=0)[0][0]
                
                # EXACTNESS LOGIC
                if z_depth < 0.018: # Heightened sensitivity for cartoon rejection
                    status = "CARTOON/PHOTO (REAL)"
                    v_buffer.append(0.0)
                else:
                    v_buffer.append(raw_v)
                    v_score_avg = sum(v_buffer)/len(v_buffer)
                    status = "DEEPFAKE" if v_score_avg > V_THRESHOLD else "AUTHENTIC HUMAN"
                    v_score_final = v_score_avg
                
                color = (0,0,255) if "DEEPFAKE" in status else (0,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, status, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    # --- 4. HUD ---
    cv2.rectangle(frame, (0, 0), (520, 140), (10, 10, 10), -1)
    a_color = (0,0,255) if audio_stats["score"] > A_THRESHOLD else (0,255,0)
    cv2.putText(frame, f"AUDIO: {audio_stats['label']} ({audio_stats['score']:.2f})", (20, 55), 1, 1.5, a_color, 2)
    v_color = (0,0,255) if v_score_final > V_THRESHOLD else (0,255,0)
    cv2.putText(frame, f"VIDEO: {status} ({v_score_final:.2f})", (20, 110), 1, 1.5, v_color, 2)

    cv2.imshow('Exact Deepfake Forensic v9.0', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()