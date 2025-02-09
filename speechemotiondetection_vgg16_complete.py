"""SpeechEmotionDetection_VGG16_Complete.ipynb

## Import required libraries
"""

#!pip install numpy pandas tqdm seaborn matplotlib scikit-learn tensorflow

import os
import librosa
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import librosa.display
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Model Traing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
# Model Evaluation
from sklearn.metrics import confusion_matrix
from tensorflow.keras import models as tf_models
from sklearn.metrics import classification_report

"""## Mounting Dataset from Google Drive"""

from google.colab import drive
drive.mount('/content/drive')

dataset_root = r"/content/drive/MyDrive"
dataset_subfolders = ["Audio_Speech_Actors_01-24"]

# Define emotions based on RAVDESS dataset encoding
emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
               '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}

#ls /content/drive/MyDrive/

# Ensuring dataset_features directory exists
output_dir = r"/content/drive/MyDrive/SpeechEmoDetection/dataset_features"
os.makedirs(output_dir, exist_ok=True)

"""## Feature Extraction"""

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Extract MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    # Extract Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

    # Extract Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

    return np.hstack([mfccs, chroma, mel])

"""## Dataset Processing"""

# Function to process a dataset folder
def process_dataset(dataset_path, category, data):
    all_files = []
    for actor_folder in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        for file_name in os.listdir(actor_path):
            file_path = os.path.join(actor_path, file_name)
            all_files.append((file_path, file_name))

    print(f"Processing {category}: {len(all_files)} files")
    for file_path, file_name in tqdm(all_files, desc=f"{category}"):
        try:
            features = extract_features(file_path)
            emotion_code = file_name.split("-")[2]
            emotion = emotion_map.get(emotion_code, "unknown")
            data.append([file_path, category, emotion] + features.tolist())
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

data = []

for subfolder in dataset_subfolders:
    process_dataset(os.path.join(dataset_root, subfolder), subfolder, data)

if data:
    columns = ["file_path", "category", "emotion"] + [f"feature_{i}" for i in range(len(data[0]) - 3)]
    df = pd.DataFrame(data, columns=columns)

    #Converting Dataframe to csv
    df.to_csv(os.path.join(output_dir, "audio_features.csv"), index=False)
    print("Feature extraction completed! Data saved to dataset_features/audio_features.csv")
else:
    print("No valid audio files found for feature extraction.")

df = pd.read_csv(r"/content/drive/MyDrive/SpeechEmoDetection/dataset_features/audio_features.csv")
df.head()

"""## Plot the Sample Waveform and MFCC"""

audio_file = random.choice(df['file_path'])

# Load audio file
y, sr = librosa.load(audio_file, sr=None)

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

# Create a figure with subplots
fig, ax = plt.subplots(nrows=2, figsize=(10, 6))

# Plot waveform
ax[0].set_title(f"Waveform of {audio_file.split('/')[-1]}")
librosa.display.waveshow(y, sr=sr, ax=ax[0])
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Amplitude")

# Plot MFCC features
img = librosa.display.specshow(mfccs, x_axis="time", sr=sr, ax=ax[1], cmap="viridis")
ax[1].set_title("MFCC Features")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("MFCC Coefficients")
fig.colorbar(img, ax=ax[1], format="%+2.f")
# Show plot
plt.tight_layout()
plt.show()

"""## Preporcessing Dataset"""

df.head()

# Encode emotions
print(df['emotion'])
label_encoder = LabelEncoder()
df["emotion"] = label_encoder.fit_transform(df["emotion"])
df['emotion']

df['emotion'].value_counts()

# Excluding non-feature columns
exclude_columns = ["emotion", "file_path", "category"]

# Split data into features and labels
X = df.drop(columns=exclude_columns, axis=1)
y = df["emotion"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=28)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

print("Data preprocessing completed! Datasets saved in dataset_features/")

"""## Model Training"""

# Load dataset
dataset_dir = r"/content/drive/MyDrive/SpeechEmoDetection/dataset_features"
X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

# Define CNN Model
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(128, kernel_size=3, activation="relu"),
    MaxPooling1D(2),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(np.unique(y_train)), activation="softmax")
])

# Compile Model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train Model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save("speech_emotion_model.h5")
print("Model trained and saved!")

"""## Model Evaluation"""

# Reshape input
X_test = np.expand_dims(X_test, axis=2)

# Load Saved model
model = tf_models.load_model("speech_emotion_model.h5")

# Predict Output
y_pred = np.argmax(model.predict(X_test), axis=1)

# Evaluate Print
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()







"""# **Speech Emotion Detection using VGG16**
This notebook provides an end-to-end implementation of **Speech Emotion Detection** using the **RAVDESS dataset** and a **VGG16-based deep learning model**.

**Pipeline:**
- Load and preprocess dataset
- Extract Mel Spectrograms as images (224x224)
- Train a VGG16-based model
- Evaluate model performance
- Save and test the trained model
"""

# Install necessary dependencies (if not installed)
#!pip install --upgrade librosa tensorflow numpy pandas tqdm seaborn matplotlib scikit-learn opencv-python

import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# Set dataset path (Change this to your dataset directory)
DATASET_PATH = r"/content/drive/MyDrive/Audio_Speech_Actors_01-24"

# Define emotions based on RAVDESS dataset encoding
emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
               '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}

# Function to extract Mel Spectrogram as 2D image for VGG16
def extract_mel_spectrogram(file_path, img_size=(224, 224)):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize to VGG16 input size (224x224)
        mel_spec_resized = cv2.resize(mel_spec_db, img_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to range [0,1]
        mel_spec_resized = (mel_spec_resized - mel_spec_resized.min()) / (mel_spec_resized.max() - mel_spec_resized.min())

        # Convert to 3-channel image for VGG16 (RGB-like format)
        mel_spec_rgb = np.stack([mel_spec_resized] * 3, axis=-1)

        return mel_spec_rgb
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset and extract features
data = []
labels = []

# Traverse dataset directory and process each file
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            file_parts = file.split("-")

            if len(file_parts) > 2:  # Ensure correct file format
                emotion_code = file_parts[2]
                if emotion_code in emotion_map:
                    spectrogram = extract_mel_spectrogram(file_path)
                    if spectrogram is not None:
                        data.append(spectrogram)
                        labels.append(int(emotion_code) - 1)  # Convert emotion to index

# Convert to NumPy arrays
X = np.array(data)
y = to_categorical(np.array(labels), num_classes=8)  # One-hot encoding for 8 emotions

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape}, Validation samples: {X_val.shape}")

# Define VGG16-based model

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(8, activation="softmax")(x)

# Define final model
vgg_model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model
vgg_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Model summary
vgg_model.summary()

# Train the model

EPOCHS = 100
BATCH_SIZE = 32

history = vgg_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save trained model
vgg_model.save("speech_emotion_vgg16_model.h5")

print("Model training complete. Saved as speech_emotion_vgg16_model.h5")

# Plot training history

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# Evaluate model with confusion matrix

y_pred = np.argmax(vgg_model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_map.values(), yticklabels=emotion_map.values())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_true, y_pred, target_names=emotion_map.values()))

#ls /content/drive/MyDrive/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav

# Load the trained model and test with a sample audio file

sample_audio_path = "/content/drive/MyDrive/Audio_Speech_Actors_01-24/Actor_01/03-01-02-01-01-01-01.wav"
sample_spectrogram = extract_mel_spectrogram(sample_audio_path)
sample_spectrogram = np.expand_dims(sample_spectrogram, axis=0)

# Load trained model
loaded_model = tf.keras.models.load_model("speech_emotion_vgg16_model.h5")

# Make prediction
prediction = loaded_model.predict(sample_spectrogram)
predicted_label = np.argmax(prediction)

# Convert predicted_label to string key format
predicted_label_str = str(predicted_label + 1).zfill(2)

print(f"Predicted Emotion: {emotion_map[predicted_label_str]}")

vgg_model.save("/content/drive/MyDrive/SpeechEmoDetection/speech_emotion_model_vgg16.h5")
print("Model trained and saved!")



import streamlit as st

model = tf.keras.models.load_model("speech_emotion_model_vgg16.h5", compile=False)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Function to extract Mel Spectrogram as a 2D image for VGG16
def extract_mel_spectrogram(audio_bytes, img_size=(224, 224)):
    audio_stream = io.BytesIO(audio_bytes)
    with wave.open(audio_stream, 'rb') as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize to VGG16 input size (224x224)
    mel_spec_resized = cv2.resize(mel_spec_db, img_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to range [0,1]
    mel_spec_resized = (mel_spec_resized - mel_spec_resized.min()) / (mel_spec_resized.max() - mel_spec_resized.min())

    # Convert to 3-channel image for VGG16 (RGB-like format)
    mel_spec_rgb = np.stack([mel_spec_resized] * 3, axis=-1)

    return np.expand_dims(mel_spec_rgb, axis=0)  # Add batch dimension

# Streamlit UI
st.title(" 🗣️ Speech Emotion Detection")

# File Upload Option
st.write("---")
st.write("📂 **Upload an audio file** for emotion detection.")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file:
    sample_audio_path = uploaded_file.read()
    sample_spectrogram = extract_mel_spectrogram(sample_audio_path)
    sample_spectrogram = np.expand_dims(sample_spectrogram, axis=0)

    # Load trained model
    loaded_model = tf.keras.models.load_model("speech_emotion_vgg16_model.h5")

    # Make prediction
    prediction = loaded_model.predict(sample_spectrogram)
    predicted_label = np.argmax(prediction)

    # Convert predicted_label to string key format
    predicted_label_str = str(predicted_label + 1).zfill(2)

    #print(f"Predicted Emotion: {emotion_map[predicted_label_str]}")
    st.write(f"Predicted Emotion: **{emotion_map.get(predicted_label_str, 'Unknown')}**")



