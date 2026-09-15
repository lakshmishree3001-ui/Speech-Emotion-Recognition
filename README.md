# 🎙️ SpeakSense — Speech Emotion Recognition System

An AI-powered Speech Emotion Recognition (SER) web application that detects human emotional states from spoken audio using Machine Learning and Deep Learning, trained on the **RAVDESS** dataset.

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Artificial Intelligence · Speech Processing · Affective Computing |
| **Dataset** | RAVDESS — 1,440 speech audio clips · 8 emotions · 24 professional actors |
| **Emotion Classes** | Neutral · Calm · Happy · Sad · Angry · Fearful · Disgust · Surprised |
| **Acoustic Features** | 264-dimensional feature vector (MFCCs, Deltas, Mel-Filterbanks, Chroma, Spectral Scalars) |
| **Primary Models** | SVM (Optimized), ANN, Mel-CNN, Random Forest |
| **Application UI** | Streamlit interactive web application (`app/app.py`) |

---

## 🏆 Model Performance Summary

| Model | Architecture / Paradigm | Accuracy | F1-Score (Weighted) | Optimal Use Case |
|-------|-------------------------|----------|---------------------|------------------|
| **SVM (Optimized)** | RBF Support Vector Classifier | **62.2%** | **62.7%** | Low-latency live microphone inference |
| **ANN (Best)** | Deep Multilayer Perceptron | **62.9%** | **62.9%** | High-accuracy static audio evaluation |
| **ANN (Optimized)** | Regularized Dense MLP | **61.8%** | **61.7%** | Noise-resilient neural classification |
| **Mel-CNN** | 2D Spatial Convolutional Network | **58.9%** | **53.1%** | Formant & time-frequency pattern analysis |
| **Random Forest** | 300-Estimator Ensemble | **49.7%** | **49.1%** | Benchmark baseline & interpretable trees |

---

## 🔄 Acoustic Feature Extraction & Inference Pipeline

```
Audio Input (Microphone / WAV / MP3 / OGG / FLAC)
    ↓
Librosa Audio Loader (Sample rate = 22,050 Hz, Mono)
    ↓
Audio Conditioning & Voice Activity Detection (VAD)
  • Dynamic silence trimming (top_db thresholding)
  • Peak normalization & 16-bit PCM amplification
  • Standardized 3.0-second windowing (66,150 samples)
    ↓
264-Dimensional Feature Extraction
  • 40 MFCCs (Spectral envelope)
  • 40 Delta MFCCs (Velocity rate)
  • 40 Delta-Delta MFCCs (Acceleration)
  • 128 Mel Filterbank Energies (Cochlear frequency bands)
  • 12 Chroma Coefficients (Pitch class distribution)
  • 4 Spectral Scalars (ZCR, Spectral Centroid, Bandwidth, Rolloff)
    ↓
StandardScaler Transform (models/scaler.pkl)
    ↓
Model Inference → Softmax / Decision Probabilities (8 classes)
    ↓
Argmax → Predicted Emotion Label + Confidence Score (%)
    ↓
Internal Telemetry Tracking (Latency profiling & performance auditing)
```

---

## 🗂 Project Structure

```
Speech Emotion Recognition/
│
├── app/
│   └── app.py                  ← Streamlit multi-view application
│
├── src/
│   ├── __init__.py
│   ├── audio_preprocessing.py  ← Audio loading, VAD & normalization
│   ├── feature_extraction.py   ← 264-D acoustic feature extraction
│   ├── monitoring.py           ← Thread-safe internal session telemetry & profiling
│   └── predictor.py            ← Unified inference engine & model manager
│
├── models/
│   ├── ann_model.keras         ← Deep Multilayer Perceptron
│   ├── ann_optimized.keras     ← Regularized ANN
│   ├── mel_cnn_best.keras      ← Mel-spectrogram CNN
│   ├── mel_cnn_config.json     ← Mel-CNN architecture configuration
│   ├── svm_optimized.pkl       ← RBF Support Vector Machine pipeline
│   ├── rf_optimized.pkl        ← Random Forest ensemble
│   ├── scaler.pkl              ← Feature StandardScaler
│   ├── label_encoder.pkl       ← 8-class emotion LabelEncoder
│   └── model_manifest.json     ← SHA-256 integrity hashes and metadata
│
├── features/
│   ├── features_dataset.csv    ← Pre-extracted 264-D feature matrix
│   └── dataset_metadata.csv    ← Audio clip metadata
│
├── notebooks/                  ← End-to-end development & evaluation notebooks
│   ├── 03_dataset_exploration.ipynb
│   ├── 04_audio_preprocessing.ipynb
│   ├── 05_exploratory_analysis.ipynb
│   ├── 06_feature_extraction.ipynb
│   ├── 07_ml_classification.ipynb
│   ├── 08_deep_learning.ipynb
│   ├── 09_model_evaluation.ipynb
│   ├── 10_realtime_prediction.ipynb
│   ├── 11_model_optimization.ipynb
│   └── 12_advanced_audio_classification.ipynb
│
├── reports/                    ← Evaluation dashboards, ROC curves & confusion matrices
├── tests/                      ← Automated unit & integration tests
│   ├── test_invalid_audio.py
│   ├── test_realtime_prediction.py
│   └── test_deployment.py
│
├── .streamlit/
│   └── config.toml             ← UI theme, server config & upload limits
│
├── packages.txt                ← Linux system packages (libsndfile1, ffmpeg)
├── requirements.txt            ← Pinned runtime dependencies
├── Dockerfile                  ← Container configuration
├── DEPLOYMENT.md               ← Deployment manual & Docker instructions
└── README.md                   ← Project documentation
```

---

## 🚀 Local Setup & Execution

### 1 — Prerequisites

- **Python 3.10, 3.11, or 3.12**
- Git installed on your system

### 2 — Create & Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Windows (PowerShell / Command Prompt)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will launch locally at **`http://localhost:8501`**.

---

## 🧭 Application Views

The interface offers a streamlined experience organized into four primary views:

1. **Home**: High-level overview of the SpeakSense architecture, pipeline workflow, emotion taxonomy, and benchmark metrics.
2. **Classify**: Interactive emotion classification workspace. Record live speech directly via microphone or upload an audio file (`.wav`, `.mp3`, `.ogg`, `.flac`) to inspect predicted emotions, confidence gauges, and acoustic parameters.
3. **Dashboards**: Deep analytical telemetry including interactive time-domain waveforms, log-mel spectrograms, emotion probability distributions, and technical model comparison matrices.
4. **About**: Scientific context covering the RAVDESS dataset, acoustic stack breakdown, feature extraction theory, and developer credits.

> [!NOTE]
> Performance monitoring, latency profiling, and system health checks run automatically and securely in the background via internal telemetry threads.

---

## 🧪 Automated Testing Suite

Run the automated test suite to verify audio preprocessing, model loading, and real-time prediction pipelines:

```bash
# Run all unit tests
python -m unittest discover -s tests -p "test_*.py" -v

# Run individual test modules
python -m unittest tests/test_invalid_audio.py -v
python -m unittest tests/test_realtime_prediction.py -v
python tests/test_deployment.py
```

---

## 📦 Key Technologies

| Technology | Purpose |
|------------|---------|
| `Streamlit` | Modern reactive web UI and client routing |
| `Librosa` | Audio loading, silence trimming, mel-spectrograms, MFCC extraction |
| `SoundFile` | Multi-format audio I/O backend |
| `Scikit-Learn` | Classical machine learning (SVM, Random Forest), feature scaling |
| `TensorFlow` / `Keras` | Deep neural network inference (ANN, Mel-CNN) |
| `Plotly` | Interactive charts, probability radars, and audio visualization |
| `NumPy` & `Pandas` | High-performance tensor manipulation and feature storage |

---

## 📄 Deployment Guide

For containerization with **Docker** or hosting on cloud platforms, refer to the [Deployment Manual](DEPLOYMENT.md).

---

*SpeakSense · Speech Emotion Recognition System*
