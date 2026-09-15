"""
src/audio_preprocessing.py
──────────────────────────
Reusable preprocessing utilities for the Speech Emotion Recognition project.
These functions are used by both the notebooks and the Streamlit app.
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf

# ── Default constants (match Module 4 notebook) ──────────────────────────────
TARGET_SR       = 22050
TARGET_DURATION = 3.0
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)   # 66,150
TOP_DB          = 30

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

EMOTION_CLASSES = list(EMOTION_MAP.values())   # index 0–7


# ── Core preprocessing pipeline ──────────────────────────────────────────────

def load_audio(filepath: str, sr: int = None) -> tuple:
    """
    Load a WAV file using librosa.

    Parameters
    ----------
    filepath : str   Path to .wav file.
    sr       : int   Target sample rate. None = load at native rate.

    Returns
    -------
    (signal: np.ndarray, sample_rate: int)
    """
    y, sr_out = librosa.load(filepath, sr=sr)
    return y, sr_out


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert a multi-channel signal to mono by averaging channels."""
    if signal.ndim > 1:
        return librosa.to_mono(signal)
    return signal


def resample(signal: np.ndarray, orig_sr: int,
             target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample a signal to target_sr if needed."""
    if orig_sr != target_sr:
        return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)
    return signal


def trim_silence(signal: np.ndarray, top_db: int = TOP_DB) -> np.ndarray:
    """Remove leading/trailing silence below `top_db` threshold."""
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return trimmed


def peak_normalize(signal: np.ndarray) -> np.ndarray:
    """Scale signal so its peak absolute amplitude equals 1.0."""
    mx = np.max(np.abs(signal))
    if mx > 0:
        return signal / mx
    return signal


def pad_or_truncate(signal: np.ndarray,
                    target_len: int = TARGET_SAMPLES) -> np.ndarray:
    """
    Centre-pad with zeros if shorter than target_len,
    or truncate from the end if longer.
    Returns exactly target_len samples.
    """
    n = len(signal)
    if n < target_len:
        pad  = target_len - n
        signal = np.pad(signal, (pad // 2, pad - pad // 2), mode='constant')
    elif n > target_len:
        signal = signal[:target_len]
    return signal


def preprocess_audio(filepath: str,
                     target_sr: int      = TARGET_SR,
                     target_samples: int = TARGET_SAMPLES,
                     top_db: int         = TOP_DB) -> np.ndarray:
    """
    Full preprocessing pipeline for a single audio file.

    Steps:
        1. Load at native sample rate
        2. Convert to mono
        3. Resample to target_sr
        4. Trim silence
        5. Peak normalise
        6. Pad or truncate to target_samples

    Returns
    -------
    np.ndarray of shape (target_samples,), dtype float32
    """
    y, sr = librosa.load(filepath, sr=None)
    y     = to_mono(y)
    y     = resample(y, orig_sr=sr, target_sr=target_sr)
    y     = trim_silence(y, top_db=top_db)
    y     = peak_normalize(y)
    y     = pad_or_truncate(y, target_len=target_samples)
    return y.astype(np.float32)


# ── RAVDESS filename parser ───────────────────────────────────────────────────

def parse_ravdess_filename(filepath: str) -> dict:
    """
    Extract metadata from a RAVDESS filename.

    Format: MM-VV-EE-II-SS-RR-AA.wav
    Returns a dict with all fields plus derived 'emotion' and 'gender'.
    """
    fname  = os.path.basename(filepath)
    parts  = os.path.splitext(fname)[0].split('-')
    actor  = int(parts[6])
    ec     = parts[2]

    return {
        'filepath'    : filepath,
        'filename'    : fname,
        'actor'       : actor,
        'gender'      : 'male' if actor % 2 == 1 else 'female',
        'emotion_code': ec,
        'emotion'     : EMOTION_MAP.get(ec, 'unknown'),
        'intensity'   : 'normal' if parts[3] == '01' else 'strong',
        'statement'   : int(parts[4]),
        'repetition'  : int(parts[5]),
        'modality'    : 'audio' if parts[0] == '03' else 'other',
        'vocal_channel': 'speech' if parts[1] == '01' else 'song'
    }


# ── Audio info helpers ────────────────────────────────────────────────────────

def get_audio_info(filepath: str) -> dict:
    """Return basic metadata about an audio file using soundfile."""
    info = sf.info(filepath)
    return {
        'format'     : info.format,
        'subtype'    : info.subtype,
        'channels'   : info.channels,
        'sample_rate': info.samplerate,
        'duration'   : round(info.duration, 3),
        'frames'     : info.frames
    }


# ── Config persistence ────────────────────────────────────────────────────────

def save_preprocessing_config(path: str,
                               target_sr: int       = TARGET_SR,
                               target_duration: float = TARGET_DURATION,
                               target_samples: int  = TARGET_SAMPLES,
                               top_db: int          = TOP_DB,
                               emotion_classes: list = None) -> None:
    """Save preprocessing config as JSON."""
    config = {
        'target_sr'       : target_sr,
        'target_duration' : target_duration,
        'target_samples'  : target_samples,
        'top_db'          : top_db,
        'emotion_classes' : emotion_classes or EMOTION_CLASSES,
        'num_classes'     : len(emotion_classes or EMOTION_CLASSES)
    }
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_preprocessing_config(path: str) -> dict:
    """Load preprocessing config from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


# ── Audio validation for deployment & inference (Module 14) ───────────────────

def validate_audio_signal(
    signal: np.ndarray,
    sr: int = TARGET_SR,
    min_duration: float = 0.25,
    min_peak: float = 1e-5,
) -> tuple[bool, str]:
    """
    Validate an audio signal array for inference.

    Parameters
    ----------
    signal       : np.ndarray Audio samples
    sr           : int Sampling rate
    min_duration : float Minimum duration in seconds
    min_peak     : float Minimum peak amplitude to reject pure silence

    Returns
    -------
    (is_valid: bool, error_message: str)
    """
    if signal is None or len(signal) == 0:
        return False, "Audio signal is empty or None."

    if not isinstance(signal, np.ndarray):
        return False, f"Expected numpy ndarray, got {type(signal).__name__}."

    if np.isnan(signal).any():
        return False, "Audio signal contains NaN (Not-a-Number) values."

    if np.isinf(signal).any():
        return False, "Audio signal contains infinite values."

    duration = len(signal) / sr
    if duration < min_duration:
        return False, (
            f"Audio duration is too short ({duration:.2f}s). "
            f"Minimum duration required is {min_duration:.2f}s."
        )

    peak = float(np.max(np.abs(signal)))
    if peak < min_peak:
        return False, (
            "Audio signal is completely silent. "
            "Please check microphone levels and speak clearly."
        )

    return True, "Valid"

