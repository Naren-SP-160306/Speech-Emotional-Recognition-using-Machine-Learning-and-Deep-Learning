import os
import numpy as np
import librosa
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ravdess_path = input("Enter path to RAVDESS dataset: ")
tess_path = input("Enter path to TESS dataset: ")
sampling_rate = 22050
n_mels = 128
n_mfcc = 128
max_time_steps = 128
hop_length = 512

def resize_feature(feature, target_shape=(128, 128)):
    if feature.shape[0] != target_shape[0]:
        feature = cv2.resize(feature, (feature.shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    if feature.shape[1] < target_shape[1]:
        feature = np.pad(feature, ((0, 0), (0, target_shape[1] - feature.shape[1])), mode='constant')
    else:
        feature = feature[:, :target_shape[1]]
    return feature

def extract_features(file_path, dataset_type):
    try:
        y, sr = librosa.load(file_path, sr=sampling_rate)
        y, _ = librosa.effects.trim(y)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        chan1 = librosa.power_to_db(mel, ref=np.max)
        chan2 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        chan3 = chroma
        chan1, chan2, chan3 = [resize_feature(ch) for ch in [chan1, chan2, chan3]]
        return np.stack([chan1, chan2, chan3], axis=-1)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_datasets():
    features, labels = [], []
    rav_emotions = {'01':'neutral', '02':'calm', '03':'happy', '04':'sad', '05':'angry', '06':'fearful', '07':'disgust', '08':'surprised'}
    print("Processing RAVDESS...")
    for root, dirs, files in os.walk(ravdess_path):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('-')
                emotion = rav_emotions.get(parts[2])
                feat = extract_features(os.path.join(root, file), 'ravdess')
                if feat is not None:
                    features.append(feat)
                    labels.append(emotion)
    print("Processing TESS...")
    for root, dirs, files in os.walk(tess_path):
        for file in files:
            if file.endswith('.wav'):
                filename_lower = file.lower()
                if 'neutral' in filename_lower or 'calm' in filename_lower:
                    emotion = 'neutral'
                elif 'angry' in filename_lower: emotion = 'angry'
                elif 'disgust' in filename_lower: emotion = 'disgust'
                elif 'fear' in filename_lower: emotion = 'fearful'
                elif 'happy' in filename_lower: emotion = 'happy'
                elif 'ps' in filename_lower or 'surprised' in filename_lower: emotion = 'surprised'
                elif 'sad' in filename_lower: emotion = 'sad'
                else: continue
                feat = extract_features(os.path.join(root, file), 'tess')
                if feat is not None:
                    features.append(feat)
                    labels.append(emotion)
    X = np.array(features)
    y = np.array(labels)
    np.save('X_resnet_features.npy', X)
    np.save('y_resnet_labels.npy', y)
    print(f"Extraction Complete! Shape: {X.shape}")

def res_block(x, filters, dropout_rate):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x

def cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = res_block(x, 64, 0.2)
    x = res_block(x, 128, 0.3)
    x = res_block(x, 256, 0.4)
    x = res_block(x, 512, 0.4)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    process_datasets()
    X = np.load('X_resnet_features.npy')
    y = np.load('y_resnet_labels.npy')
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    model = cnn_model((128, 128, 3), num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
    print("Starting Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[lr_reducer, early_stop]
    )
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")