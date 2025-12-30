import os
import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
import threading
from tensorflow.keras.models import load_model

# --- 1. SETUP & HARDWARE CHECK ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide TF warnings

try:
    video_model = load_model('deepfake_video_detector.h5')
    audio_model = load_model('deepfake_audio_detector.h5')
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Model Error: Ensure .h5 files are in this folder. {e}")
    exit()

# Global variables for results
audio_label = "WAITING..."
audio_score = 0.0

# --- 2. AUDIO PROCESSING THREAD ---
def audio_monitor_thread():
    global audio_label, audio_score
    fs = 16000
    duration = 1.0 # 1 second chunks
    
    print("🎙️ Audio Thread Started...")
    while True:
        try:
            # Capture 1s of audio
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            audio_data = recording.flatten()
            
            # Create Spectrogram (128 x 109)
            # hop_length=148 at 16k SR creates exactly ~109 time steps
            S = librosa.feature.melspectrogram(y=audio_data, sr=fs, n_mels=128, hop_length=148)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Force size to exactly 109 width to avoid ValueError
            if S_dB.shape[1] < 109:
                S_dB = np.pad(S_dB, ((0, 0), (0, 109 - S_dB.shape[1])), mode='constant')
            else:
                S_dB = S_dB[:, :109]

            # Prepare for AI: (Batch, Height, Width, Channels)
            input_data = S_dB.reshape(1, 128, 109, 1)
            
            prediction = audio_model.predict(input_data, verbose=0)
            audio_score = float(prediction[0][0])
            audio_label = "FAKE VOICE" if audio_score > 0.5 else "REAL VOICE"
            
        except Exception as e:
            audio_label = "MIC ERROR"
            print(f"Audio Error: {e}")

# Launch Audio Thread
threading.Thread(target=audio_monitor_thread, daemon=True).start()

# --- 3. VIDEO PROCESSING LOOP ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Video Stream Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Process Frame for AI
    # Resize to 224x224 (typical for deepfake video models)
    small_frame = cv2.resize(frame, (224, 224))
    v_input = np.expand_dims(small_frame, axis=0) / 255.0
    
    v_score = video_model.predict(v_input, verbose=0)[0][0]
    v_label = "FAKE FACE" if v_score > 0.5 else "REAL FACE"

    # --- 4. THE UI DASHBOARD ---
    # Colors (BGR format): Red for Fake, Green for Real
    color_v = (0, 0, 255) if v_score > 0.5 else (0, 255, 0)
    color_a = (0, 0, 255) if audio_score > 0.5 else (0, 255, 0)
    
    # Overlay Background Box
    cv2.rectangle(frame, (10, 10), (450, 130), (0, 0, 0), -1)
    
    # Text Labels
    cv2.putText(frame, f"VIDEO: {v_label} ({v_score:.2f})", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_v, 2)
    cv2.putText(frame, f"AUDIO: {audio_label} ({audio_score:.2f})", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_a, 2)
    
    # Critical Alert
    if v_score > 0.7 and audio_score > 0.7:
        cv2.putText(frame, "!! DEEPFAKE DETECTED !!", (100, 400), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow('Deepfake Detector v1.0', frame)

    # Required for Windows to process the window display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()