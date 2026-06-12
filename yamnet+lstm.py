import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import sounddevice as sd
import soundfile as sf
import os

VALID_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
print(" Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print(" Loading emotion model...")
emotion_model_path = input("Enter path to emotion_best_model.keras file: ")
model = tf.keras.models.load_model(emotion_model_path)
print(" Models loaded!")
def predict_from_file(file_path):
    print(f"\nLoading audio: {file_path}")
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    features = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)
    predictions = model.predict(features, verbose=0)
    predicted_idx = np.argmax(predictions)
    confidence = predictions[0][predicted_idx] * 100
    print(f"\n Predicted Emotion : {VALID_LABELS[predicted_idx].upper()}")
    print(f" Confidence : {confidence:.2f}%")
    print("\n All Emotions:")
    for i, label in enumerate(VALID_LABELS):
         bar = "█" * int(predictions[0][i] * 20)
        print(f" {label:10s}: {bar} {predictions[0][i]*100:.1f}%")

def record_and_predict(duration=5):
    print(f"\nRecording for {duration} seconds...")
    print("Speak now!")
    recording = sd.rec(
        int(duration * 16000),
        samplerate=16000,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print(" Recording done!")
    temp_path = "temp_recording.wav"
    sf.write(temp_path, recording, 16000)
    predict_from_file(temp_path)

print("\n================================")
print(" EMOTION DETECTION SYSTEM")
print("================================")
print("1. Predict from audio file")
print("2. Record and predict")
print("================================")
choice = input("Enter choice (1 or 2): ")
if choice == "1":
    path = input("Enter full path of .wav file: ")
    predict_from_file(path)
elif choice == "2":
    record_and_predict(duration=5)
else:
    print("Invalid choice!")