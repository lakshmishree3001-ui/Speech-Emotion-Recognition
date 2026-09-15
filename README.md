# 🎙️ SpeakSense — Speech Emotion Recognition System

AI-powered speech emotion classification using Machine Learning and Deep Learning,
trained on the **RAVDESS** dataset.

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Artificial Intelligence · Speech Processing |
| **Dataset** | RAVDESS — 1 440 clips · 8 emotions · 24 actors |
| **Emotions** | Neutral · Calm · Happy · Sad · Angry · Fearful · Disgust · Surprised |
| **Primary Model** | ANN (`ann_model.keras`) — 62.9% accuracy |
| **Interface** | Streamlit web application (`app/app.py`) |

---

## 🏆 Model Performance Summary

| Model | Accuracy | F1 (weighted) | Type |
|-------|----------|---------------|------|
| **ANN (Best)** | **62.9%** | **62.9%** | Deep Learning |
| ANN Optimized | 61.8% | 61.7% | Deep Learning |
| SVM Optimized | 62.2% | 62.7% | Machine Learning (Pipeline) |
| Mel-CNN | 58.9% | 53.1% | Deep Learning (Spectrogram) |
| Random Forest | 49.7% | 49.1% | Machine Learning |

---

## 🔄 Inference Pipeline

```
Audio Input (WAV / MP3 / OGG / FLAC)
    ↓
Librosa load (sr=22 050 Hz, mono)
    ↓
Trim silence (top_db=30)
    ↓
Peak normalise
    ↓
Pad / truncate → 3.0 seconds (66 150 samples)
    ↓
Feature extraction  [264-dim vector]
    MFCC (40) + MFCC-delta (40) + MFCC-delta² (40)
    + Mel filterbank energies (128)
    + Chroma (12)
    + ZCR, Spectral Centroid, Bandwidth, Rolloff (4)
    ↓
StandardScaler transform (models/scaler.pkl)   ← ANN only
    ↓
Model inference → class probabilities (8 classes)
    ↓
Argmax → predicted emotion label
    ↓
Confidence score (%)
```

---

## 🗂 Project Structure

```
Speech Emotion Recognition/
│
├── app/
│   └── app.py                  ← Streamlit application (main entry point)
│
├── src/
│   ├── __init__.py
│   ├── audio_preprocessing.py  ← Audio loading & preprocessing
│   ├── feature_extraction.py   ← 264-dim feature extractor
│   └── predictor.py            ← Inference backend
│
├── models/
│   ├── ann_model.keras         ← Best ANN model (62.9% acc)
│   ├── ann_optimized.keras     ← Optimized ANN
│   ├── mel_cnn_best.keras      ← Mel-spectrogram CNN
│   ├── mel_cnn_config.json     ← Mel-CNN configuration
│   ├── svm_optimized.pkl       ← SVM Pipeline (with internal scaler)
│   ├── rf_optimized.pkl        ← Random Forest
│   ├── scaler.pkl              ← StandardScaler (for ANN)
│   └── label_encoder.pkl       ← LabelEncoder (8 emotion classes)
│
├── Data/
│   └── Raw/
│       └── Audio_Speech_Actors_01-24/   ← RAVDESS dataset
│
├── features/
│   ├── features_dataset.csv    ← Pre-extracted feature matrix
│   └── dataset_metadata.csv    ← Audio clip metadata
│
├── notebooks/                  ← Training & evaluation notebooks
│   ├── 06_feature_extraction.ipynb
│   ├── 07_ml_classification.ipynb
│   ├── 08_deep_learning.ipynb
│   ├── 09_model_evaluation.ipynb
│   ├── 10_realtime_prediction.ipynb
│   ├── 11_model_optimization.ipynb
│   └── 12_advanced_audio_classification.ipynb
│
├── reports/                    ← Evaluation charts & CSVs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Local Setup & Run

### 1 — Prerequisites

- Python 3.9 or higher
- A virtual environment (recommended)

### 2 — Create & activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the application

```bash
streamlit run app/app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🌐 Production Deployment & Real-Time Monitoring (Module 14)

SpeakSense is fully pre-configured and hardened for cloud deployment on **Streamlit Community Cloud** or **Docker / Containers**.

For detailed, step-by-step instructions, see the complete [Deployment Manual](DEPLOYMENT.md).

### Quick Deployment Steps
1. **Push to GitHub**: Initialize git, commit all files (ensuring `Data/Raw/` is excluded via `.gitignore`), and push to GitHub.
2. **Deploy on Streamlit Cloud**: Go to [streamlit.io/cloud](https://streamlit.io/cloud), choose your repository, select `app/app.py` as the main entrypoint, and click **Deploy**.
3. **Verify Deployment Health**: Open the **Monitoring** tab in the top navigation to verify model health status (`🟢 HEALTHY`), latency profiling, and manifest integrity.

> **Deployment Assets Included:**
> - `packages.txt` — Linux system dependencies (`libsndfile1`, `ffmpeg`).
> - `.streamlit/config.toml` — Production server settings, CORS, and upload limits.
> - `Dockerfile` & `.dockerignore` — Containerized deployment option.
> - `models/model_manifest.json` — SHA-256 cryptographic hashes & model parameter specs.
> - `src/monitoring.py` — In-memory session telemetry, latency profiling, and audit logging.
> - `tests/` — Automated test suites for invalid audio handling, real-time prediction, and deployment readiness.

---

## 🧪 Automated Testing Suite

Run the full automated test suite using Python:

```bash
# 1. Invalid audio handling & resilience tests
python -m unittest tests/test_invalid_audio.py -v

# 2. Real-time audio prediction & latency benchmarks
python -m unittest tests/test_realtime_prediction.py -v

# 3. Deployment preflight readiness & manifest verification
python tests/test_deployment.py

# 4. Optional: Check live deployed endpoint
python tests/test_deployment.py --url https://<your-app>.streamlit.app
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `librosa` | Audio loading & feature extraction |
| `soundfile` | Audio I/O backend |
| `scikit-learn` | ML models · StandardScaler · LabelEncoder |
| `tensorflow` | Keras deep learning models |
| `numpy` | Numerical computation |
| `pandas` | Data handling |
| `matplotlib` | Waveform & spectrogram visualisations |
| `plotly` | Interactive dashboard & monitoring charts |
| `psutil` | System resource & telemetry monitoring |

---

## 👩‍💻 Usage

1. Open the app in your browser (`http://localhost:8501`).
2. Explore views from the top navigation bar:
   - **Home**: System architecture, workflow, and model performance overview.
   - **Classify**: Live speech emotion prediction via microphone recording, audio upload, or verified RAVDESS samples.
   - **Dashboards**: Deep dive analytics, model comparison, feature correlations, and audio exploratory data.
   - **Monitoring**: Live operational telemetry, latency timeline, model manifest checksums, and audit logs.
   - **About**: Dataset details, emotion taxonomy, and technical citations.

---

*Module 14 · Model Deployment & Real-Time Monitoring · Speech Emotion Recognition System*

