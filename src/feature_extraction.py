"""
src/feature_extraction.py
─────────────────────────
Reusable feature extraction utilities for the Speech Emotion Recognition project.
Used by notebooks, training scripts, and the Streamlit application.
"""

import numpy as np
import librosa

# ── Default constants ─────────────────────────────────────────────────────────
N_MFCC     = 40
N_MELS     = 128
N_CHROMA   = 12
N_FFT      = 2048
HOP_LENGTH = 512
FMAX       = 8000
SR         = 22050

# Total feature dimensions:
#   MFCC(40) + MFCC_delta(40) + MFCC_delta2(40) + Mel(128) + Chroma(12) + Scalars(4) = 264
N_FEATURES = N_MFCC * 3 + N_MELS + N_CHROMA + 4


def extract_features(signal: np.ndarray, sr: int = SR,
                     n_mfcc: int = N_MFCC, n_mels: int = N_MELS,
                     n_chroma: int = N_CHROMA, n_fft: int = N_FFT,
                     hop_length: int = HOP_LENGTH, fmax: int = FMAX) -> np.ndarray:
    """
    Extract a 264-dimensional feature vector from a preprocessed audio signal.

    Feature Groups
    --------------
    - MFCC (40)           : Mel-Frequency Cepstral Coefficients, mean over time
    - MFCC delta (40)     : First-order temporal derivative of MFCC
    - MFCC delta2 (40)    : Second-order temporal derivative of MFCC
    - Mel Spectrogram (128): Log-power Mel filterbank energies, mean over time
    - Chroma (12)         : Pitch class energy distribution, mean over time
    - ZCR (1)             : Zero Crossing Rate, mean over time
    - Spectral Centroid (1): Weighted mean frequency, mean over time
    - Spectral Bandwidth (1): Spread of spectrum, mean over time
    - Spectral Rolloff (1) : Frequency below 85% energy, mean over time

    Parameters
    ----------
    signal     : np.ndarray  Preprocessed audio signal (float32, 1-D)
    sr         : int          Sample rate (default 22050 Hz)
    n_mfcc     : int          Number of MFCC coefficients (default 40)
    n_mels     : int          Number of Mel filter bands (default 128)
    n_chroma   : int          Number of chroma bins (default 12)
    n_fft      : int          FFT window size (default 2048)
    hop_length : int          Hop size in samples (default 512)
    fmax       : int          Max frequency for Mel/Chroma (default 8000 Hz)

    Returns
    -------
    np.ndarray of shape (264,), dtype float32
    """
    # ── MFCC + deltas ─────────────────────────────────────────────────────
    mfcc    = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                   n_fft=n_fft, hop_length=hop_length)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    f_mfcc  = np.concatenate([
        mfcc.mean(axis=1), mfcc_d.mean(axis=1), mfcc_d2.mean(axis=1)
    ])  # 120 values

    # ── Mel Spectrogram (log power) ───────────────────────────────────────
    mel    = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels,
                                            fmax=fmax, n_fft=n_fft,
                                            hop_length=hop_length)
    f_mel  = librosa.power_to_db(mel, ref=np.max).mean(axis=1)  # 128 values

    # ── Chroma ────────────────────────────────────────────────────────────
    chroma   = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=n_chroma,
                                            n_fft=n_fft, hop_length=hop_length)
    f_chroma = chroma.mean(axis=1)  # 12 values

    # ── Scalar features ───────────────────────────────────────────────────
    zcr     = librosa.feature.zero_crossing_rate(signal, hop_length=hop_length)
    sc      = librosa.feature.spectral_centroid(y=signal, sr=sr, hop_length=hop_length)
    sb      = librosa.feature.spectral_bandwidth(y=signal, sr=sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, hop_length=hop_length,
                                               roll_percent=0.85)
    f_scalar = np.array([
        float(zcr.mean()), float(sc.mean()),
        float(sb.mean()), float(rolloff.mean())
    ])  # 4 values

    return np.concatenate([f_mfcc, f_mel, f_chroma, f_scalar]).astype(np.float32)


def get_feature_names(n_mfcc: int = N_MFCC, n_mels: int = N_MELS,
                      n_chroma: int = N_CHROMA) -> list:
    """Return a list of feature column names matching extract_features output."""
    return (
        [f'mfcc_{i+1}'     for i in range(n_mfcc)]    +
        [f'mfcc_d_{i+1}'   for i in range(n_mfcc)]    +
        [f'mfcc_d2_{i+1}'  for i in range(n_mfcc)]    +
        [f'mel_{i+1}'      for i in range(n_mels)]     +
        [f'chroma_{i+1}'   for i in range(n_chroma)]   +
        ['zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']
    )


def extract_mfcc_only(signal: np.ndarray, sr: int = SR,
                      n_mfcc: int = N_MFCC, n_fft: int = N_FFT,
                      hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Extract only MFCC features (no delta, no other features).
    Returns mean over time frames: shape (n_mfcc,).
    Used for quick baseline experiments.
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    return mfcc.mean(axis=1).astype(np.float32)


def extract_mfcc_2d(signal: np.ndarray, sr: int = SR,
                    n_mfcc: int = N_MFCC, n_fft: int = N_FFT,
                    hop_length: int = HOP_LENGTH,
                    max_frames: int = 130) -> np.ndarray:
    """
    Extract a 2-D MFCC array for CNN input.
    Returns shape (n_mfcc, max_frames) — padded or truncated along time axis.
    Used by the CNN deep learning model.
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    # Pad or truncate time dimension
    if mfcc.shape[1] < max_frames:
        pad = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_frames]
    return mfcc.astype(np.float32)
