import os, io
import wave
import threading
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import pyaudio
import pickle

# Load the trained emotion detection model
model = tf.keras.models.load_model("dataset_features\speech_emotion_model.h5", compile=False)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
recording = False
frames = []


# Function to extract features from audio bytes
def extract_audio_features(audio_bytes):
    audio_stream = io.BytesIO(audio_bytes)
    with wave.open(audio_stream, 'rb') as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])


# Function to start recording
def start_recording():
    global recording, frames
    recording = True
    frames = []

    def record():
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK)
        while recording:
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()

    thread = threading.Thread(target=record)
    thread.start()


# Function to stop recording and process audio without saving
def stop_recording():
    global recording
    recording = False

    # Convert recorded frames to a bytes object
    audio_bytes = b''.join(frames)

    # Create an in-memory WAV file
    audio_stream = io.BytesIO()
    with wave.open(audio_stream, 'wb') as wave_file:
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(audio_bytes)

    return audio_stream.getvalue()  # Return audio bytes


# Streamlit UI
st.title("Speech Emotion Detection")

st.write("🎤 Click **Start Recording** to begin speaking, and **Stop Recording** to analyze.")

# Create Streamlit buttons
if st.button("Start Recording 🎙️"):
    start_recording()
    st.write("Recording... Speak now!")

if st.button("Stop Recording ⏹️"):
    audio_bytes = stop_recording()

    # Extract features and make prediction
    features = extract_audio_features(audio_bytes)
    features = np.expand_dims(features, axis=(0, 2))

    prediction = model.predict(features)
    emotion_label = np.argmax(prediction)

    # Emotion mapping
    emotion_map = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust",
                   7: "surprised"}
    st.write(f"Predicted Emotion: **{emotion_map.get(emotion_label, 'Unknown')}**")

# File Upload Option
st.write("---")
st.write("📂 **Upload an audio file** for emotion detection.")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file:
    audio_bytes = uploaded_file.read()

    features = extract_audio_features(audio_bytes)
    features = np.expand_dims(features, axis=(0, 2))

    prediction = model.predict(features)
    print(prediction)
    emotion_label = np.argmax(prediction)
    print(emotion_label)
    emotion_map = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust",
                   8: "surprised"}
    st.write(f"Predicted Emotion: **{emotion_map.get(emotion_label, 'Unknown')}**")
