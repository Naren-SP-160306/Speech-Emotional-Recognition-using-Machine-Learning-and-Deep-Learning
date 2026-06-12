import os
import warnings
import pandas as pd
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
confusion_matrix, roc_curve, auc,
precision_recall_curve, average_precision_score)
from sklearn.preprocessing import RobustScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

base_dir = input("Enter path to base directory: ")
tess_path = input("Enter path to TESS dataset: ")
ravdess_path = input("Enter path to RAVDESS dataset: ")
ravdess_meta_path = input("Enter path to RAVDESS metadata CSV: ")

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprised',
    'neutral': 'neutral', 'calm': 'calm', 'happy': 'happy', 'sad': 'sad',
    'angry': 'angry', 'fear': 'fear', 'fearful': 'fear', 'disgust': 'disgust',
    'ps': 'surprised', 'surprised': 'surprised'
}

def extract_features(y, sr):
    if np.max(np.abs(y)) < 1e-6:
        return np.zeros(282)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(mel, axis=1), np.std(mel, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        [np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms),
         np.mean(centroid), np.std(centroid), np.mean(rolloff), np.std(rolloff)]
    ])
    return features

def load_data():
    data = []
    if os.path.exists(ravdess_meta_path):
        try:
            rav_df = pd.read_csv(ravdess_meta_path)
            for _, row in rav_df.iterrows():
                path = row['path']
                if not os.path.exists(path):
                    path = os.path.join(ravdess_path, os.path.basename(path))
                parts = os.path.basename(path).split("-")
                emotion = emotion_map.get(parts[2]) if len(parts) >= 3 else None
                if emotion and os.path.exists(path):
                    data.append((path, emotion))
        except Exception:
            pass
    if not data:
        for root, _, files in os.walk(ravdess_path):
            for file in files:
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if len(parts) >= 3:
                        emotion = emotion_map.get(parts[2])
                        if emotion:
                            data.append((os.path.join(root, file), emotion))
    for root, _, files in os.walk(tess_path):
        for file in files:
            if file.endswith(".wav"):
                label = file.split("_")[-1].replace(".wav", "").lower()
                emotion = emotion_map.get(label)
                if emotion:
                    data.append((os.path.join(root, file), emotion))
    return pd.DataFrame(data, columns=['path', 'emotion'])

df = load_data()
if df.empty:
    raise ValueError("Dataset empty. Verify paths.")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])

def prepare_dataset(data_df, chunk_duration=2.5):
    X, Y = [], []
    sr = 22050
    chunk_length = int(sr * chunk_duration)
    for _, row in data_df.iterrows():
        try:
            y, _ = librosa.load(row['path'], sr=sr)
            y, _ = librosa.effects.trim(y, top_db=25)
            if len(y) < chunk_length:
                y = librosa.util.fix_length(y, size=chunk_length)
            num_chunks = int(np.ceil(len(y) / chunk_length))
            for i in range(num_chunks):
                start = i * chunk_length
                chunk = librosa.util.fix_length(y[start:start+chunk_length], size=chunk_length)
                X.append(extract_features(librosa.util.normalize(chunk), sr))
                Y.append(row['emotion'])
        except Exception:
            continue
    return np.array(X), np.array(Y)

X_train, Y_train = prepare_dataset(train_df)
X_test, Y_test = prepare_dataset(test_df)
np.savez(os.path.join(base_dir, 'features_cache.npz'), X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

le = LabelEncoder()
Y_train_enc = le.fit_transform(Y_train)
Y_test_enc = le.transform(Y_test)
class_names = le.classes_
cw = compute_class_weight('balanced', classes=np.unique(Y_train_enc), y=Y_train_enc)
class_weight_dict = dict(enumerate(cw))

pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('svm', SVC(C=10, kernel='rbf', probability=True, class_weight=class_weight_dict, cache_size=1000))
])

pipeline.fit(X_train, Y_train_enc)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(Y_test_enc, y_pred)*100:.2f}%")
print(classification_report(Y_test_enc, y_pred, target_names=class_names))

cm = confusion_matrix(Y_test_enc, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))
plt.show()

joblib.dump(pipeline, os.path.join(base_dir, 'emotion_model.pkl'))
joblib.dump(le, os.path.join(base_dir, 'label_encoder.pkl'))