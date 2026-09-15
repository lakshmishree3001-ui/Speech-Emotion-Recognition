# 🚀 SpeakSense — Production Deployment Manual (Module 14)

This guide provides end-to-end instructions for deploying the **SpeakSense Speech Emotion Recognition System** to **Streamlit Community Cloud** (primary) and **Docker / Cloud Containers** (alternative).

---

## 📋 Pre-Deployment Readiness Checklist

Before deploying, verify that all necessary files are in place:

| Asset | File Path | Status | Purpose |
|-------|-----------|--------|---------|
| **Streamlit Entrypoint** | `app/app.py` | ✅ Ready | Multi-view web application |
| **Python Dependencies** | `requirements.txt` | ✅ Ready | Pinned runtime packages |
| **Linux System Packages**| `packages.txt` | ✅ Ready | `libsndfile1`, `ffmpeg` for Debian |
| **Streamlit Config** | `.streamlit/config.toml`| ✅ Ready | Port, CORS, upload size (25MB) |
| **Trained Models** | `models/*.keras`, `*.pkl`| ✅ Ready | 5 trained emotion models (< 30MB each) |
| **Preprocessing Artifacts**| `models/scaler.pkl`, `label_encoder.pkl` | ✅ Ready | Normalization & class decoding |
| **Model Manifest** | `models/model_manifest.json` | ✅ Ready | Integrity hashes and metadata |
| **Raw Audio Dataset** | `Data/Raw/` | 🛡️ Ignored | Excluded by `.gitignore` (too large) |

---

## 🌐 Method 1: Deploying to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is the fastest, free way to deploy Streamlit apps with automatic CI/CD on git push.

### Step 1 — Initialize Git and Commit Your Code
Run these commands in PowerShell or Git Bash inside `c:\Speech Emotion Recognition`:

```bash
# 1. Initialize git repository (if not already done)
git init

# 2. Check git status to ensure Data/Raw/ is ignored
git status

# 3. Add all project files
git add .

# 4. Create your initial deployment commit
git commit -m "feat(module-14): prepare SpeakSense for Streamlit Cloud deployment"
```

> [!IMPORTANT]
> Verify that `Data/Raw/` is **NOT** staged in git. The models in `models/` are all under 30 MB (well below GitHub's 100 MB per-file limit) and must be included in your repository.

### Step 2 — Push to GitHub
1. Go to [github.com](https://github.com) and click **New Repository**.
2. Name it `Speech-Emotion-Recognition` (or `speaksense`).
3. Keep it **Public** (required for free Streamlit Community Cloud).
4. Run the remote push commands shown on GitHub:

```bash
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/Speech-Emotion-Recognition.git
git push -u origin main
```

### Step 3 — Deploy on Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io) (or [streamlit.io/cloud](https://streamlit.io/cloud)) and sign in with your GitHub account.
2. Click **New app** (top right).
3. Fill in the deployment form:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/Speech-Emotion-Recognition`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
   - **App URL:** (Choose a custom subdomain, e.g., `speaksense-emotion.streamlit.app`)
4. Click **Advanced settings** (optional):
   - **Python version:** Select `3.11` or `3.12`.
5. Click **Deploy!**

### Step 4 — Verify the Live Deployment
1. Streamlit will install packages from `packages.txt` (`libsndfile1`, `ffmpeg`) and `requirements.txt`.
2. Once the build completes, your app will open at `https://<your-app>.streamlit.app`.
3. In the top navigation, click **Monitoring**:
   - Verify that **System Health** displays `🟢 HEALTHY`.
   - Verify that all **5 / 5 Active Models** are loaded and manifest verified.
   - Click **⚡ Trigger Model Warmup & Benchmark** to prime models and test latency.
4. Go to **Classify**, record or upload an audio clip, and test emotion prediction live!

---

## 🐳 Method 2: Deploying via Docker (Containerized)

For self-hosting, Hugging Face Spaces, Render, Railway, AWS ECS, or GCP Cloud Run:

### 1. Build the Docker Image
```bash
docker build -t speaksense:latest .
```

### 2. Run the Container Locally
```bash
docker run -d -p 8501:8501 --name speaksense_app speaksense:latest
```

### 3. Access and Verify
Open your browser to:
```
http://localhost:8501
```
Check health endpoint:
```
http://localhost:8501/_stcore/health
```

---

## 🛠️ Troubleshooting Common Cloud Deployment Issues

### 1. Audio Decoding Fails (`libsndfile.so` not found)
- **Cause:** Streamlit Cloud Linux environment missing native audio libraries.
- **Solution:** `packages.txt` is already created with `libsndfile1` and `ffmpeg`. Ensure `packages.txt` is committed to the root of your GitHub repository.

### 2. File Too Large Git Rejection
- **Cause:** Committing large raw WAV files from `Data/Raw/`.
- **Solution:** `Data/Raw/` is listed in `.gitignore`. If accidentally staged, run:
  ```bash
  git rm -r --cached Data/Raw
  git commit -m "fix: remove raw audio dataset from git tracking"
  ```

### 3. Cold-Start Latency on First Request
- **Cause:** TensorFlow Keras and scikit-learn graph compilation.
- **Solution:** Navigate to the **Monitoring** view in the app and click **⚡ Trigger Model Warmup & Benchmark**, or let the built-in warmup initialize automatically on first startup.

### 4. Audio Input Silent or Click Transients
- **Cause:** Microphone hardware pop or user speaking too quietly.
- **Solution:** The app now features Voice Activity Detection (VAD) and graceful `SilentAudioError` / `AudioTooShortError` alerts that inform the user without crashing.

---

*Module 14 — Model Deployment & Real-Time Monitoring*
