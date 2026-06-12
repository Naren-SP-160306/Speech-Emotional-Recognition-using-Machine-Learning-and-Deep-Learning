# Speech Emotion Recognition (SER) using Machine Learning and Deep Learning

This repository contains a Speech Emotion Recognition (SER) system developed to identify human emotions from audio recordings. The project explores, implements, and compares the performance of both traditional machine learning algorithms and advanced deep learning architectures.

## 📊 Datasets Used
* **RAVDESS**: The Ryerson Audio-Visual Database of Emotional Speech and Song (24 professional actors).
* **TESS**: Toronto Emotional Speech Set (2 professional actresses).

## 🚀 Model Architectures Evaluated
We implemented and benchmarked four distinct approaches to find the most accurate emotion classification pipeline:

### 1. Deep Learning Approaches
* **CNN + ResNet**: Utilizes Convolutional Neural Networks combined with Residual Network blocks to extract complex spatial features from audio spectrograms.
* **YamNet + LSTM**: Leverages Google's pretrained YamNet model for high-level audio feature extraction, coupled with Long Short-Term Memory (LSTM) networks to capture temporal sequence dependencies.

### 2. Traditional Machine Learning Approaches
* **Support Vector Machine (SVM)**: Baseline classification using extracted statistical acoustic features (MFCCs, chroma, mel-spectrogram).
* **XGBoost**: Extreme Gradient Boosting classifier optimized for fast performance and tabular audio feature maps.

## 🛠️ Tech Stack
* **Languages**: Python
* **Frameworks**: TensorFlow, Keras, Scikit-learn, XGBoost
* **Audio Processing**: Librosa
