<p align="center">
<a href="" rel="noopener">
<img width=200px height=200px src="https://yt3.googleusercontent.com/caqRf4lIyodH_wmjdafODKqOd00fGDCjH8YXPKJxr2EFcVZTqG4ZnOx54lME3e80Q1jEjx8e=s160-c-k-c0x00ffffff-no-rj" alt="Project logo"></a>
</p>

<h3 align="center">Deepfake & Cartoon Forensic System</h3>

<div align="center">

</div>

<p align="center"> A high-precision, real-time forensic suite designed to detect AI-generated video (Deepfakes), synthetic audio, and 2D spoofing attempts using 3D biometric liveness and spectral analysis.





</p>

🧐 About <a name = "about"></a>

This project provides a robust solution for real-time digital integrity verification. By combining MediaPipe's 3D Face Mesh with a Deep Learning Audio Analysis engine, the system can distinguish between a real human, a deepfake video, and a simple 2D image or cartoon.

The core purpose is to prevent identity spoofing in video calls or recordings by checking for "Biometric Liveness" (Z-depth variance) and identifying spectral artifacts common in AI-generated synthetic speech.

🏁 Getting Started <a name = "getting_started"></a>

These instructions will help you set up the forensic suite on your local machine for real-time analysis.

Prerequisites

You need Python 3.10+ installed and a working webcam and microphone.

# Verify your python version
python --version


Installing

Navigate to the project directory:

cd "C:\Users\Administrator\OneDrive\Desktop\Python\Real time Deepfake AudioVideo Detector"


Install the required dependencies:

pip install opencv-python mediapipe numpy librosa sounddevice tensorflow scipy


Ensure Model Files are present:
The directory must contain:

deepfake_video_detector.h5

deepfake_audio_detector.h5

🎈 Usage <a name="usage"></a>

Run the application:

python main_ver_9.0.py


Forensic HUD: The screen displays a real-time status for both Audio and Video.

Biometric Calibration: Stay approximately 1-2 feet from the camera for accurate depth sensing.

Exit: Press 'q' to close the session.

📊 Logic Matrix <a name = "logic_matrix"></a>

Result

Video Logic

Audio Logic

AUTHENTIC HUMAN

Z-Depth > 0.018 + AI < 0.85

Real Harmonics

DEEPFAKE

Z-Depth > 0.018 + AI > 0.85

Synthetic Artifacts

CARTOON / PHOTO

Z-Depth < 0.018 (Flat)

Signal Ignored

⛏️ Built Using <a name = "built_using"></a>

TensorFlow - Deep Learning Framework

MediaPipe - 3D Face Mesh & Landmarks

Librosa - Audio Processing

OpenCV - Computer Vision

SoundDevice - Real-time Audio I/O

✍️ Authors <a name = "authors"></a>

Your Name / Admin - Lead Development & Optimization

Gemini AI - Engineering & Bug Resolution

🎉 Acknowledgements <a name = "acknowledgement"></a>

MediaPipe Team for the 3D Face Mesh API.

Librosa community for spectral analysis tools.

Inspiration from modern liveness detection research.
