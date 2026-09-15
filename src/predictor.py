"""
src/predictor.py
────────────────
Reusable inference backend for the Speech Emotion Recognition application.

Exact pipeline (matches training in notebooks 06–12):
  Audio File/Bytes
  → Librosa load (sr=22050, mono, duration=3.0 s)
  → Trim silence → Peak normalise → Pad/Truncate (66150 samples)
  → Feature extraction OR Mel-spectrogram (model-dependent)
  → Scaler (for ANN; SVM pipeline has internal scaler)
  → Model inference
  → Label decode (label_encoder.pkl or mel_cnn_config.json)
  → Structured result dict
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Project root (two levels up from this file: src/ → project root) ─────────
SRC_DIR   = Path(__file__).resolve().parent
BASE_DIR  = SRC_DIR.parent
MODEL_DIR = BASE_DIR / "models"

# ── Add src/ to sys.path so relative imports work when running from app/ ─────
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Audio constants (must match training) ─────────────────────────────────────
SR             = 22_050
DURATION       = 3.0
N_SAMPLES      = int(SR * DURATION)   # 66 150
TOP_DB         = 30

# Mel-spectrogram constants (from mel_cnn_config.json)
N_MELS         = 128
N_FFT          = 2_048
HOP_LENGTH     = 512
FMAX           = 8_000
MEL_TARGET_T   = 130   # time frames → (128, 130, 1)

# Feature-vector constants (from src/feature_extraction.py)
N_FEATURES     = 264

# ── Emotion metadata ──────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "neutral":   "😐",
    "calm":      "😌",
    "happy":     "😊",
    "sad":       "😢",
    "angry":     "😠",
    "fearful":   "😨",
    "disgust":   "🤢",
    "surprised": "😲",
}

EMOTION_COLOR = {
    "neutral":   "#94a3b8",
    "calm":      "#34d399",
    "happy":     "#fbbf24",
    "sad":       "#38bdf8",
    "angry":     "#f87171",
    "fearful":   "#a78bfa",
    "disgust":   "#fb923c",
    "surprised": "#f472b6",
}

EMOTION_DESC = {
    "neutral":   "The speaker has a flat, unemotional tone.",
    "calm":      "The speaker is relaxed and composed.",
    "happy":     "The speaker sounds joyful and positive.",
    "sad":       "The speaker sounds sorrowful or downcast.",
    "angry":     "The speaker sounds frustrated or intense.",
    "fearful":   "The speaker sounds anxious or scared.",
    "disgust":   "The speaker expresses strong aversion.",
    "surprised": "The speaker sounds astonished or unexpected.",
}

# ── Model performance (from evaluation reports) ───────────────────────────────
MODEL_PERFORMANCE = {
    "SVM Optimized": {"accuracy": "62.2%", "f1": "62.7%", "type": "Machine Learning"},
    "ANN (Best)":    {"accuracy": "62.8%", "f1": "62.9%", "type": "Deep Learning (Dense ANN)"},
    "ANN Optimized": {"accuracy": "61.8%", "f1": "61.7%", "type": "Deep Learning"},
    "Mel-CNN":       {"accuracy": "58.9%", "f1": "53.1%", "type": "Deep Learning (Spectrogram)"},
    "Random Forest": {"accuracy": "49.7%", "f1": "49.1%", "type": "Machine Learning"},
}


# ── Custom Exceptions for Error Handling (Module 14) ───────────────────────────
class AudioProcessingError(Exception):
    """Base exception for all audio processing and emotion inference errors."""
    pass

class SilentAudioError(AudioProcessingError):
    """Raised when the audio signal is silent or below detectable threshold."""
    pass

class AudioTooShortError(AudioProcessingError):
    """Raised when audio duration is insufficient (< 0.25s voiced speech)."""
    pass

class InvalidAudioFormatError(AudioProcessingError):
    """Raised when audio source format is empty, invalid, or unsupported."""
    pass

class CorruptedAudioError(AudioProcessingError):
    """Raised when audio file cannot be decoded or contains NaN/Inf values."""
    pass


# ════════════════════════════════════════════════════════════════════════════════
# ModelManager — loads and caches all model artifacts
# ════════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Singleton-style lazy loader and cache for all model artifacts."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._models: dict = {}
        self._scaler = None
        self._label_encoder = None
        self._mel_cfg: dict = {}
        self._loaded = False

    def _path(self, filename: str) -> Path:
        return self.model_dir / filename

    def load_all(self) -> dict:
        """Load every artifact once and cache in memory."""
        if self._loaded:
            return self._get_status()

        import joblib
        import tensorflow.keras as keras

        errors: list[str] = []

        # ── Preprocessing objects ─────────────────────────────────────────
        scaler_path = self._path("scaler.pkl")
        le_path     = self._path("label_encoder.pkl")
        mel_cfg_path= self._path("mel_cnn_config.json")

        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler.pkl  (n_features=%d)", self._scaler.n_features_in_)
        else:
            errors.append("scaler.pkl not found")

        if le_path.exists():
            self._label_encoder = joblib.load(le_path)
            logger.info("Loaded label_encoder.pkl  classes=%s", self._label_encoder.classes_)
        else:
            errors.append("label_encoder.pkl not found")

        if mel_cfg_path.exists():
            with open(mel_cfg_path, "r") as f:
                self._mel_cfg = json.load(f)
            logger.info("Loaded mel_cnn_config.json")
        else:
            errors.append("mel_cnn_config.json not found")

        # ── Keras models ──────────────────────────────────────────────────
        keras_models = {
            "ANN (Best)":    "ann_model.keras",      # 62.85% acc — best ANN
            "ANN Optimized": "ann_optimized.keras",  # 61.8% acc
            "Mel-CNN":       "mel_cnn_best.keras",   # 58.9% acc — mel-spectrogram
        }
        for name, fname in keras_models.items():
            fpath = self._path(fname)
            if fpath.exists():
                try:
                    self._models[name] = keras.models.load_model(str(fpath))
                    logger.info("Loaded %s  (%s)", name, fname)
                except Exception as e:
                    errors.append(f"{fname}: {e}")
                    logger.error("Failed to load %s: %s", fname, e)
            else:
                errors.append(f"{fname} not found")

        # ── Sklearn / Pipeline models ─────────────────────────────────────
        sklearn_models = {
            "SVM Optimized": "svm_optimized.pkl",   # Pipeline (scales internally)
            "Random Forest": "rf_optimized.pkl",    # RF — expects raw 264 features
        }
        for name, fname in sklearn_models.items():
            fpath = self._path(fname)
            if fpath.exists():
                try:
                    self._models[name] = joblib.load(str(fpath))
                    logger.info("Loaded %s  (%s)", name, fname)
                except Exception as e:
                    errors.append(f"{fname}: {e}")
                    logger.error("Failed to load %s: %s", fname, e)
            else:
                errors.append(f"{fname} not found")

        self._loaded = True
        if errors:
            logger.warning("ModelManager load warnings: %s", errors)
        return self._get_status()

    def _get_status(self) -> dict:
        return {
            "models_available":  list(self._models.keys()),
            "scaler_ready":      self._scaler is not None,
            "le_ready":          self._label_encoder is not None,
            "mel_cfg_ready":     bool(self._mel_cfg),
        }

    def warmup_models(self) -> dict:
        """
        Execute dummy inference through all loaded models to eliminate initial cold-start latency.
        """
        if not self._loaded:
            self.load_all()
        warmup_results = {}
        dummy_flat = np.zeros((1, N_FEATURES), dtype=np.float32)
        dummy_mel = np.zeros((1, N_MELS, MEL_TARGET_T, 1), dtype=np.float32)

        for name, model in self._models.items():
            t0 = time.perf_counter()
            try:
                if name == "Mel-CNN":
                    _ = model.predict(dummy_mel, verbose=0)
                elif name in ("ANN (Best)", "ANN Optimized"):
                    if self._scaler is not None:
                        scaled = self._scaler.transform(dummy_flat)
                        _ = model.predict(scaled, verbose=0)
                    else:
                        _ = model.predict(dummy_flat, verbose=0)
                else:
                    _ = model.predict_proba(dummy_flat)
                ms = (time.perf_counter() - t0) * 1000
                warmup_results[name] = {"status": "READY", "warmup_ms": round(ms, 2)}
            except Exception as e:
                warmup_results[name] = {"status": "WARMUP_FAILED", "error": str(e)}
        return warmup_results

    def get_health_status(self) -> dict:
        """Comprehensive health status check for deployment monitoring."""
        status = self._get_status()
        manifest_path = self._path("model_manifest.json")
        status["manifest_present"] = manifest_path.exists()
        status["total_models_loaded"] = len(self._models)
        status["healthy"] = (
            len(self._models) >= 3
            and self._scaler is not None
            and self._label_encoder is not None
        )
        return status

    @property
    def models(self) -> dict:
        if not self._loaded:
            self.load_all()
        return self._models

    @property
    def scaler(self):
        if not self._loaded:
            self.load_all()
        return self._scaler

    @property
    def label_encoder(self):
        if not self._loaded:
            self.load_all()
        return self._label_encoder

    @property
    def mel_cfg(self) -> dict:
        if not self._loaded:
            self.load_all()
        return self._mel_cfg


# ════════════════════════════════════════════════════════════════════════════════
# Voice Activity Detection & Audio Preprocessing
# ════════════════════════════════════════════════════════════════════════════════

def detect_voice_activity(
    signal: np.ndarray,
    sr: int = SR,
    hop_length: int = 512,
    top_db: int = 28,
) -> tuple[int, int, bool, float, float]:
    """
    Robust Voice Activity Detection (VAD) that rejects short click transients (< 60ms)
    and extracts continuous speech segments.

    Returns:
        (start_sample, end_sample, is_speech, voiced_duration_sec, mean_rms)
    """
    if len(signal) == 0:
        return 0, 0, False, 0.0, 0.0

    import librosa
    # Remove DC offset so baseline is centered at 0.0
    signal = signal - np.mean(signal)
    sig_peak = float(np.max(np.abs(signal))) if len(signal) > 0 else 0.0
    if sig_peak < 1e-6:
        return 0, len(signal), False, 0.0, 0.0

    # Work with peak-normalized signal so VAD is invariant to hardware gain / mic level
    sig_norm = signal / sig_peak
    rms = librosa.feature.rms(y=sig_norm, hop_length=hop_length)[0]
    if len(rms) == 0:
        return 0, 0, False, 0.0, 0.0

    p95 = float(np.percentile(rms, 95))
    mean_rms = float(np.mean(rms))

    thresh = max(0.015, p95 * (10 ** (-top_db / 20.0)))
    active = rms > thresh

    # Require at least 3 consecutive frames (~70ms) to ignore 3ms clicks
    min_frames = 3
    runs = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if (i - start) >= min_frames:
                runs.append((start, i))
            start = None
    if start is not None and (len(active) - start) >= min_frames:
        runs.append((start, len(active)))

    if not runs:
        # Fallback: if audio has audible energy, treat as speech
        if sig_peak > 1e-4:
            return 0, len(signal), True, len(signal) / sr, mean_rms * sig_peak
        return 0, len(signal), False, 0.0, mean_rms * sig_peak

    total_active_frames = sum(e - s for s, e in runs)
    voiced_duration = total_active_frames * hop_length / sr

    # Margin of 0.15s (6 frames)
    margin = int(0.15 * sr / hop_length)
    first_start = max(0, (runs[0][0] - margin) * hop_length)
    last_end = min(len(signal), (runs[-1][1] + margin) * hop_length)

    is_speech = (voiced_duration >= 0.15)
    return first_start, last_end, is_speech, voiced_duration, mean_rms * sig_peak


def preprocess_input_audio(
    source,
    sr: int = SR,
    duration: float = DURATION,
    top_db: int = TOP_DB,
) -> tuple[np.ndarray, float]:
    """
    Load and preprocess audio from a file path, bytes, or file-like object.

    Pipeline (matches training — src/audio_preprocessing.py):
      1. Load at native SR → mono
      2. Resample to 22 050 Hz
      3. Voice Activity Detection (VAD) & silence trimming (rejects clicks)
      4. Peak normalise
      5. Pad (centre) or fit energetic window to 66 150 samples (3.0 s)

    Parameters
    ----------
    source  : str | bytes | BytesIO | np.ndarray
    sr      : target sample rate (22 050)
    duration: clip length in seconds (3.0)
    top_db  : silence trim threshold in dB (30)

    Returns
    -------
    (signal: np.ndarray float32, actual_duration_sec: float)

    Raises
    ------
    ValueError  – audio is silent or contains no voice
    RuntimeError – failed to decode the source
    """
    import librosa
    from src.audio_preprocessing import (
        to_mono, resample, peak_normalize, pad_or_truncate
    )

    n_target = int(sr * duration)

    try:
        # ── Resolve source ────────────────────────────────────────────────
        y_native = None
        sr_native = sr

        if isinstance(source, np.ndarray):
            y_native = source.astype(np.float32)
            raw_bytes = None
        elif isinstance(source, (str, Path)):
            with open(str(source), "rb") as f:
                raw_bytes = f.read()
        elif isinstance(source, bytes):
            raw_bytes = source
        elif hasattr(source, "read"):
            raw_bytes = source.read()
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # ── Decode audio into numpy float array (supports WAV, WebM, MP3, OGG) ──
        if y_native is None and raw_bytes is not None:
            if len(raw_bytes) == 0:
                raise InvalidAudioFormatError("Audio source is empty (0 bytes). Please upload a valid audio recording.")

            # Method A: Try soundfile first (fast for WAV/FLAC)
            try:
                import soundfile as sf
                data, native_sr = sf.read(io.BytesIO(raw_bytes))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                y_native = data.astype(np.float32)
                sr_native = native_sr
            except Exception:
                pass

            # Method B: Universal decoder using imageio_ffmpeg (handles WebM, Opus, MP3, etc.)
            if y_native is None:
                try:
                    import subprocess, soundfile as sf, imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    cmd = [
                        ffmpeg_exe, "-y",
                        "-i", "pipe:0",
                        "-vn",
                        "-ac", "1",
                        "-ar", str(sr),
                        "-f", "wav",
                        "pipe:1"
                    ]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = proc.communicate(input=raw_bytes)
                    if proc.returncode == 0 and len(out) > 0:
                        data, native_sr = sf.read(io.BytesIO(out))
                        y_native = data.astype(np.float32)
                        sr_native = native_sr
                except Exception as e:
                    logger.warning("FFmpeg decode fallback encountered error: %s", e)

            # Method C: Last-resort fallback with librosa / tempfile
            if y_native is None:
                import tempfile, librosa
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = tmp.name
                    data, native_sr = librosa.load(tmp_path, sr=None, mono=True)
                    y_native = data.astype(np.float32)
                    sr_native = native_sr
                except Exception as e:
                    logger.warning("Librosa fallback decode failed: %s", e)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

        if y_native is None and raw_bytes is not None and len(raw_bytes) == 0:
            raise InvalidAudioFormatError("Audio source is empty (0 bytes). Please upload a valid audio recording.")

        if y_native is None or len(y_native) == 0:
            raise CorruptedAudioError("Could not decode audio data from the input source. File may be corrupted or unsupported.")

        if np.isnan(y_native).any() or np.isinf(y_native).any():
            raise CorruptedAudioError("Audio signal contains invalid numeric values (NaN or Infinity).")

        actual_dur = len(y_native) / sr_native
        if actual_dur < 0.25:
            raise AudioTooShortError(
                f"Audio duration is too short ({actual_dur:.2f}s). "
                f"Minimum 0.25 seconds of speech is required."
            )

        # ── Full preprocessing chain (matches training) ───────────────────
        y = to_mono(y_native)
        y = resample(y, orig_sr=sr_native, target_sr=sr)

        # Remove DC offset
        y = y - np.mean(y)

        # Suppress startup hardware pop on microphone opening (first 20ms)
        fade_samples = min(int(0.02 * sr), len(y))
        if fade_samples > 0:
            y[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        peak_val = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak_val < 1e-6:
            raise SilentAudioError(
                "Audio signal is completely silent. "
                "Please verify that your microphone is unmuted in Windows Sound Settings and speak clearly."
            )

        # Voice activity detection to reject silence/clicks and find speech bounds
        s_idx, e_idx, is_speech, v_dur, m_rms = detect_voice_activity(y, sr=sr, top_db=top_db)
        if is_speech and (e_idx - s_idx) >= int(0.25 * sr):
            y = y[s_idx:e_idx]
        else:
            logger.info("VAD detected faint speech or room tone; proceeding with full normalized signal")

        # Peak normalize
        y = peak_normalize(y)

        # Fit to exactly n_target (66150 samples) using center-padding or energetic window
        n = len(y)
        if n < n_target:
            pad = n_target - n
            y = np.pad(y, (pad // 2, pad - pad // 2), mode='constant')
        elif n > n_target:
            # Find the 3.0s window with the highest vocal energy to preserve intonation
            step = 512 * 4
            best_start = 0
            max_energy = -1.0
            for s in range(0, n - n_target + 1, step):
                chunk = y[s:s + n_target]
                energy = float(np.sum(chunk**2))
                if energy > max_energy:
                    max_energy = energy
                    best_start = s
            y = y[best_start:best_start + n_target]

        # ── Sanity checks ─────────────────────────────────────────────────
        if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
            raise SilentAudioError(
                "Audio appears to be silent after preprocessing. "
                "Please provide a clear speech recording."
            )

        return y.astype(np.float32), actual_dur

    except Exception:
        raise


# ════════════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ════════════════════════════════════════════════════════════════════════════════

def extract_flat_features(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extract the exact 264-dimensional feature vector used during training.

    Uses src.feature_extraction.extract_features (identical to notebooks 06–11).

    Returns
    -------
    np.ndarray of shape (264,), dtype float32
    """
    from src.feature_extraction import extract_features
    return extract_features(signal, sr=sr)   # (264,)


def extract_mel_spectrogram_tensor(
    signal: np.ndarray,
    sr: int = SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmax: int = FMAX,
    target_t: int = MEL_TARGET_T,
) -> np.ndarray:
    """
    Compute a log-Mel spectrogram tensor for the Mel-CNN model.

    Processing (matches finish_nb12.py / notebook 12):
      1. librosa melspectrogram  → (n_mels, T) power spectrogram
      2. power_to_db             → log scale
      3. min-max normalise to [0, 1]
      4. Pad or truncate time axis to target_t (130)
      5. Add channel dimension   → (1, n_mels, target_t, 1)

    Returns
    -------
    np.ndarray of shape (1, 128, 130, 1), dtype float32
    """
    import librosa

    S   = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length, fmax=fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    # Pad or truncate time axis
    if S_db.shape[1] < target_t:
        S_db = np.pad(S_db, ((0, 0), (0, target_t - S_db.shape[1])))
    else:
        S_db = S_db[:, :target_t]

    return S_db[np.newaxis, ..., np.newaxis].astype(np.float32)   # (1, 128, 130, 1)


# ════════════════════════════════════════════════════════════════════════════════
# Core Prediction Function
# ════════════════════════════════════════════════════════════════════════════════

def predict_emotion(
    signal: np.ndarray,
    model_name: str,
    manager: ModelManager,
    actual_duration: float = DURATION,
) -> dict:
    """
    Run the full inference pipeline and return a structured result dict.

    Parameters
    ----------
    signal       : preprocessed audio signal (66 150 samples, float32)
    model_name   : one of the keys in ModelManager.models
    manager      : a loaded ModelManager instance
    actual_duration: duration of the original (unpadded) audio clip

    Returns
    -------
    dict with keys:
        predicted_emotion  : str
        confidence         : float  (0–100)
        probabilities      : np.ndarray  (8,)
        emotion_labels     : list[str]   (ordered to match probabilities)
        model_name         : str
        inference_ms       : float
        audio_duration     : float
        feature_shape      : str
        error              : None | str
    """
    result = {
        "predicted_emotion": None,
        "confidence":        None,
        "probabilities":     None,
        "emotion_labels":    None,
        "model_name":        model_name,
        "inference_ms":      None,
        "audio_duration":    actual_duration,
        "feature_shape":     None,
        "error":             None,
    }

    model = manager.models.get(model_name)
    if model is None:
        result["error"] = f"Model '{model_name}' is not loaded."
        return result

    t0 = time.perf_counter()

    try:
        # ── Mel-CNN branch ────────────────────────────────────────────────
        if model_name == "Mel-CNN":
            mel_cfg   = manager.mel_cfg
            n_mels    = mel_cfg.get("n_mels",    N_MELS)
            n_fft_    = mel_cfg.get("n_fft",     N_FFT)
            hop_len_  = mel_cfg.get("hop_length", HOP_LENGTH)
            fmax_     = mel_cfg.get("fmax",       FMAX)
            tgt_t     = mel_cfg.get("input_shape", [n_mels, MEL_TARGET_T, 1])[1]

            x = extract_mel_spectrogram_tensor(
                signal, sr=SR,
                n_mels=n_mels, n_fft=n_fft_, hop_length=hop_len_,
                fmax=fmax_, target_t=tgt_t,
            )
            result["feature_shape"] = str(x.shape)

            probs = model.predict(x, verbose=0)[0]

            em_map    = mel_cfg.get("emotion_to_idx", {})
            idx_to_em = {v: k for k, v in em_map.items()}
            # Sort by index to build ordered label list
            em_labels = [idx_to_em[i] for i in range(len(probs))]

        # ── Flat-feature branch (ANN, SVM, RF) ───────────────────────────
        else:
            feat = extract_flat_features(signal)          # (264,)
            result["feature_shape"] = f"({len(feat)},)"

            feat_2d = feat.reshape(1, -1)                 # (1, 264)

            if model_name in ("ANN (Best)", "ANN Optimized"):
                # ANN expects StandardScaler-normalised features
                if manager.scaler is None:
                    raise RuntimeError("scaler.pkl is not loaded — cannot run ANN.")
                feat_scaled = manager.scaler.transform(feat_2d)
                probs = model.predict(feat_scaled, verbose=0)[0]
                le = manager.label_encoder
                em_labels = list(le.classes_) if le is not None else [str(i) for i in range(len(probs))]

            elif model_name == "SVM Optimized":
                # svm_optimized.pkl is a Pipeline that includes its own StandardScaler
                probs = model.predict_proba(feat_2d)[0]
                le = manager.label_encoder
                em_labels = list(le.classes_) if le is not None else [str(i) for i in range(len(probs))]

            elif model_name == "Random Forest":
                # rf_optimized expects raw (unscaled) 264-dim features
                probs = model.predict_proba(feat_2d)[0]
                le = manager.label_encoder
                em_labels = list(le.classes_) if le is not None else [str(i) for i in range(len(probs))]

            else:
                raise ValueError(f"Unknown model: {model_name}")

    except Exception as exc:
        result["error"] = str(exc)
        logger.exception("Prediction failed for model '%s'", model_name)
        try:
            from src.monitoring import tracker
            tracker.record_prediction(
                model_name=model_name,
                predicted_emotion="error",
                confidence=0.0,
                latency_ms=0.0,
                audio_duration_sec=actual_duration,
                error=str(exc),
            )
        except Exception:
            pass
        return result

    elapsed_ms = (time.perf_counter() - t0) * 1000

    top_idx              = int(np.argmax(probs))
    result["predicted_emotion"] = em_labels[top_idx]
    result["confidence"]        = float(probs[top_idx]) * 100.0
    result["probabilities"]     = probs
    result["emotion_labels"]    = em_labels
    result["inference_ms"]      = elapsed_ms

    # ── Log to session telemetry ──────────────────────────────────────
    try:
        from src.monitoring import tracker
        tracker.record_prediction(
            model_name=model_name,
            predicted_emotion=result["predicted_emotion"],
            confidence=result["confidence"],
            latency_ms=elapsed_ms,
            audio_duration_sec=actual_duration,
            error=None,
        )
    except Exception as e:
        logger.debug("Failed to record telemetry: %s", e)

    return result


# ════════════════════════════════════════════════════════════════════════════════
# Convenience one-shot function (for quick scripting/testing)
# ════════════════════════════════════════════════════════════════════════════════

def predict_from_file(filepath: str | Path, model_name: str = "ANN (Best)") -> dict:
    """
    End-to-end convenience wrapper: load audio → preprocess → predict.

    Parameters
    ----------
    filepath   : path to a WAV / MP3 / OGG / FLAC audio file
    model_name : model to use (default 'ANN (Best)')

    Returns
    -------
    Full prediction result dict (same structure as predict_emotion).
    """
    manager = ModelManager()
    manager.load_all()

    signal, actual_dur = preprocess_input_audio(filepath)
    return predict_emotion(signal, model_name, manager, actual_duration=actual_dur)
