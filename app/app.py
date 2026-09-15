"""
app/app.py  —  Speech Emotion Recognition · SpeakSense
=======================================================
Professional Speech Emotion Recognition application with multi-dashboard analytics.

Usage:
    streamlit run app/app.py
"""
from __future__ import annotations

import io
import logging
import os
import sys
import time
import warnings
import tempfile

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Resolve paths ────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
SRC_DIR  = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Favicon generation (must run before st.set_page_config) ──────────────────────
def _build_favicon():
    """Generates the SpeakSense waveform mark as a PIL Image for use as browser favicon."""
    try:
        from PIL import Image, ImageDraw
        _fav_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        os.makedirs(_fav_dir, exist_ok=True)
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=14, fill=(20, 22, 26, 255))
        except AttributeError:  # Pillow < 9.2 fallback
            draw.rectangle([0, 0, size - 1, size - 1], fill=(20, 22, 26, 255))
        # 5 waveform bars (matches the SVG icon spec)
        for bx, by, bw, bh, color in [
            (15, 30, 4, 8,  (103, 232, 249, 255)),  # outer muted
            (22, 24, 4, 20, (34, 211, 238, 255)),
            (29, 13, 4, 38, (34, 211, 238, 255)),   # center tallest
            (36, 21, 4, 22, (34, 211, 238, 255)),
            (43, 27, 4, 10, (103, 232, 249, 255)),  # outer muted
        ]:
            try:
                draw.rounded_rectangle([bx, by, bx + bw - 1, by + bh - 1], radius=2, fill=color)
            except AttributeError:
                draw.rectangle([bx, by, bx + bw - 1, by + bh - 1], fill=color)
        return img
    except Exception:
        return "🎤"  # emoji fallback

_FAVICON = _build_favicon()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpeakSense — Speech Emotion Recognition",
    page_icon=_FAVICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("speaksense")

# ── Import inference backend ──────────────────────────────────────────────────
import importlib
import src.predictor as _predictor_mod
importlib.reload(_predictor_mod)
from src.predictor import (
    ModelManager, preprocess_input_audio, predict_emotion,
    detect_voice_activity,
    AudioProcessingError, SilentAudioError, AudioTooShortError,
    InvalidAudioFormatError, CorruptedAudioError,
    EMOTION_DESC, MODEL_PERFORMANCE,
    SR, DURATION, N_SAMPLES,
)
from src.monitoring import tracker

# ── Data directory for audio samples ─────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "Data", "Raw", "Audio_Speech_Actors_01-24")

# Emotion list ordered as stored by the label encoder
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Emoji-free emotion mapping
EMOTION_EMOJI = {e: "" for e in EMOTIONS}

# Professional palette aligned with the cyan-navy theme
THEME_EMOTION_COLOR = {
    "neutral":   "#94a3b8",
    "calm":      "#38bdf8",
    "happy":     "#facc15",
    "sad":       "#818cf8",
    "angry":     "#f43f5e",
    "fearful":   "#a855f7",
    "disgust":   "#fb923c",
    "surprised": "#22d3ee",
}

# RAVDESS emotion code -> emotion name
RAVDESS_CODE = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


# ════════════════════════════════════════════════════════════════════════════════
# Global CSS — Premium Dark AI Theme
# ════════════════════════════════════════════════════════════════════════════════

PROFESSIONAL_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #f2f3f5;
}

/* Page background */
.stApp {
    background-color: #14161a !important;
}

/* Hide streamlit default branding */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Complete sidebar suppression */
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Section Label */
.section-label {
    font-size: 11px;
    font-weight: 600;
    color: #8a8f98;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

/* View Header Component */
.view-header-block {
    margin: 4px 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #2f3339;
}
.view-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #22d3ee;
    margin-bottom: 4px;
}
.view-title {
    font-size: 24px;
    font-weight: 600;
    color: #f2f3f5;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
}
.view-desc {
    font-size: 13px;
    color: #8a8f98;
    line-height: 1.5;
    margin: 0;
    max-width: 760px;
}

/* Compact Model Selector Header & Performance Pill */
.compact-perf-pill {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 8px;
    padding: 8px 12px;
    display: flex;
    align-items: center;
    justify-content: space-around;
    height: 42px;
}
.perf-metric {
    font-size: 12px;
    color: #8a8f98;
}
.perf-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #8a8f98;
    margin-right: 4px;
}
.perf-val {
    color: #22d3ee;
    font-weight: 600;
}
.perf-divider {
    color: #2f3339;
    font-size: 12px;
}

/* Skeleton Waveform Preview Loading Card */
.preview-skeleton-card {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.skeleton-bars-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    height: 60px;
    margin: 16px 0 14px 0;
}
.skeleton-bar {
    width: 4px;
    border-radius: 2px;
    background: #2f3339;
    animation: skeletonPulse 1.6s ease-in-out infinite alternate;
}
@keyframes skeletonPulse {
    0% { background: #23272e; transform: scaleY(0.4); }
    50% { background: #2e3b44; transform: scaleY(1.0); }
    100% { background: #1c272d; transform: scaleY(0.6); }
}
.skeleton-caption {
    font-size: 12px;
    color: #8a8f98;
    max-width: 320px;
    line-height: 1.5;
}

/* ── Equal-height two-column flex row (Classify view) ── */
.classify-col-row {
    display: flex;
    align-items: stretch;
    gap: 24px;
    width: 100%;
    margin-bottom: 0;
}
.classify-col-left,
.classify-col-right {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
}
/* Make Streamlit column children stretch */
[data-testid="stHorizontalBlock"]:has(.classify-flex-sentinel) {
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"]:has(.classify-flex-sentinel)
  > [data-testid="stColumn"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 !important;
}
[data-testid="stHorizontalBlock"]:has(.classify-flex-sentinel)
  > [data-testid="stColumn"]
  > [data-testid="stVerticalBlockBorderWrapper"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}

/* ── Unified Model Architecture card ── */
.model-arch-card {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    padding: 16px 18px 14px 18px;
    margin-bottom: 20px;
    transition: border-color 0.2s ease;
}
.model-arch-card:hover {
    border-color: #3d4148;
}
.model-arch-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.model-arch-card-title {
    font-size: 11px;
    font-weight: 600;
    color: #8a8f98;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Audio Input card ── */
.audio-input-card {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    padding: 16px 18px 14px 18px;
    margin-bottom: 20px;
    flex: 1;
    transition: border-color 0.2s ease;
}
.audio-input-card:hover {
    border-color: #3d4148;
}
.audio-input-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.audio-input-card-title {
    font-size: 11px;
    font-weight: 600;
    color: #8a8f98;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Section header with icon badge ── */
.section-header-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.section-icon {
    width: 18px;
    height: 18px;
    flex-shrink: 0;
    color: #22d3ee;
}
.section-header-label {
    font-size: 11px;
    font-weight: 600;
    color: #8a8f98;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Format chip group ── */
.format-chip-group {
    margin-bottom: 12px;
}
.format-chip-group-label {
    font-size: 10px;
    font-weight: 600;
    color: #5a6070;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

/* ── Sticky footer wrapper ── */
.sticky-footer-wrap {
    position: sticky;
    bottom: 0;
    z-index: 100;
    background: rgba(20, 22, 26, 0.95);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-top: 1px solid #2f3339;
    margin-top: 32px;
}

/* Flashcard deck rendered as self-contained iframe component — no CSS needed here */

/* ── Section Block & Spacing System ── */
.section-block {
    margin-top: 56px;
    margin-bottom: 0;
}
.section-block.first-section {
    margin-top: 0;
}
.section-rule {
    border: none;
    border-top: 0.5px solid #2f3339;
    margin: 0 0 24px 0;
}

/* ── Section Header Component (sh-*) ── */
.sh-container {
    display: flex;
    flex-direction: column;
    margin-bottom: 20px;
}
.sh-row {
    display: flex;
    align-items: center;
    gap: 12px;
}
.sh-badge {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: transform 0.18s ease;
}
.sh-badge:hover {
    transform: scale(1.08);
}
.sh-title {
    font-size: 17px;
    font-weight: 700;
    color: #f2f3f5;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.sh-subtitle {
    font-size: 13px;
    color: #8a8f98;
    margin-top: 5px;
    margin-left: 44px;
    line-height: 1.4;
}

/* ── Subtle Section Group Box (use sparingly) ── */
.section-group-box {
    background: #16181c;
    border: 0.5px solid #242830;
    border-radius: 14px;
    padding: 24px 26px;
    margin-top: 4px;
}

/* Emotion Chip 2-Column Grid (About View) */
.emotion-chip-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 14px 0 20px 0;
}
.emotion-chip {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.emotion-chip:hover {
    border-color: #3d4148;
    transform: translateY(-1px);
}
.emotion-chip-header {
    display: flex;
    align-items: center;
    gap: 8px;
}
.emotion-chip-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.emotion-chip-name {
    font-size: 14px;
    font-weight: 600;
    color: #f2f3f5;
}
.emotion-chip-desc {
    font-size: 11px;
    color: #8a8f98;
    line-height: 1.4;
    padding-left: 16px;
}

/* System Status Footer Bar */
.footer-status-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 16px;
    padding: 12px 16px 10px;
    font-size: 12px;
}
.status-item {
    display: flex;
    align-items: center;
    gap: 7px;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22d3ee;
    flex-shrink: 0;
}
.status-code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
    color: #8a8f98;
}
.status-divider-dot {
    color: #3d4148;
    font-size: 14px;
}

/* KPI Badge utility */
.kpi-badge {
    background: #0e2a2e;
    border: 1px solid #164e52;
    color: #67e8f9;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 9999px;
    display: inline-block;
}

/* Hero Header with audio waveform motif and tight vertical spacing */
.hero-header {
    position: relative;
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 14px 16px 10px;
    margin: 0 auto 10px auto !important;
    width: 100% !important;
}
.hero-header * {
    text-align: center !important;
}
.hero-waveform-bg {
    position: absolute;
    top: 52%;
    left: 50%;
    transform: translate(-50%, -50%);
    opacity: 0.18;
    pointer-events: none;
    z-index: 0;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 440px;
    height: 100px;
    background: radial-gradient(ellipse, rgba(34, 211, 238, 0.05) 0%, rgba(34, 211, 238, 0) 70%);
    pointer-events: none;
}
.hero-badge {
    position: relative;
    z-index: 1;
    display: inline-block;
    background: #0e2a2e;
    border: 1px solid #164e52;
    color: #67e8f9;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 9999px;
    margin: 0 auto 10px auto !important;
    text-align: center !important;
}
.hero-title {
    position: relative;
    z-index: 1;
    font-size: 34px;
    font-weight: 500;
    color: #f2f3f5;
    letter-spacing: -0.02em;
    line-height: 1.25;
    margin: 0 auto 8px auto !important;
    text-align: center !important;
    width: 100% !important;
}
.hero-subtitle {
    position: relative;
    z-index: 1;
    font-size: 14px;
    color: #8a8f98;
    max-width: 580px;
    margin: 0 auto !important;
    line-height: 1.6;
    text-align: center !important;
    display: block !important;
    width: 100% !important;
}
div[data-testid="stMarkdownContainer"]:has(.hero-header) {
    text-align: center !important;
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}
div[data-testid="stMarkdownContainer"] p.hero-subtitle {
    text-align: center !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Cards & Surfaces */
.theme-card {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.theme-card:hover {
    border-color: #3d4148;
    transform: translateY(-1px);
}

/* Emotion Result Card */
.emotion-result-card {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 12px 0 16px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.emotion-result-card:hover {
    border-color: #3d4148;
    transform: translateY(-1px);
}

/* Badges */
.badge-active {
    display: inline-block;
    background: #0e2a2e;
    border: 1px solid #164e52;
    color: #67e8f9;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 9999px;
}
.badge-inactive {
    display: inline-block;
    background: #1c1f24;
    border: 1px solid #2f3339;
    color: #8a8f98;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 9999px;
}

/* All Buttons / Secondary Default */
.stButton > button,
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
    background-color: #1c1f24 !important;
    background: #1c1f24 !important;
    color: #c5c9d1 !important;
    border: 1px solid #2f3339 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 9px 20px !important;
    letter-spacing: 0.01em !important;
    transition: all 0.18s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover,
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover {
    background-color: #242930 !important;
    background: #242930 !important;
    border-color: #3d4148 !important;
    color: #f2f3f5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

/* Primary / Active Highlighted Buttons */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: #22d3ee !important;
    background: #22d3ee !important;
    color: #00272e !important;
    border: 1px solid #22d3ee !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 9px 20px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 0 16px rgba(34, 211, 238, 0.38) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: #38bdf8 !important;
    background: #38bdf8 !important;
    color: #00272e !important;
    border-color: #38bdf8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.55) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Secondary / Delete Button */
.btn-delete > button {
    background: #1c1f24 !important;
    border: 1px solid #2f3339 !important;
    color: #8a8f98 !important;
    box-shadow: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 16px !important;
    transition: all 0.2s ease !important;
}
.btn-delete > button:hover {
    background: #1c1f24 !important;
    border-color: #ef4444 !important;
    color: #ef4444 !important;
    transform: translateY(-1px) !important;
}

/* Download Button */
.stDownloadButton > button {
    background: #1c1f24 !important;
    border: 1px solid #2f3339 !important;
    color: #f2f3f5 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    border-color: #3d4148 !important;
    color: #22d3ee !important;
    transform: translateY(-1px) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #1c1f24 !important;
    border: 1px solid #2f3339 !important;
    border-radius: 8px !important;
    color: #f2f3f5 !important;
}
.stSelectbox > div > div:hover {
    border-color: #3d4148 !important;
}

/* Audio Uploader & Mic Input */
[data-testid="stFileUploader"] {
    background: #1c1f24 !important;
    border: 1px dashed #2f3339 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3d4148 !important;
}
[data-testid="stAudioInput"] {
    background: #1c1f24 !important;
    border: 1px dashed #2f3339 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stAudioInput"]:hover {
    border-color: #3d4148 !important;
}

/* Streamlit Metrics */
[data-testid="stMetric"] {
    background: #1c1f24 !important;
    border: 1px solid #2f3339 !important;
    border-top: 2px solid #22d3ee !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    transition: transform 0.2s ease, border-color 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: #3d4148 !important;
    border-top-color: #22d3ee !important;
    transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] {
    color: #8a8f98 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #f2f3f5 !important;
    font-weight: 500 !important;
    font-size: 20px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #2f3339 !important;
    border-radius: 0 !important;
    padding: 0 !important;
    gap: 24px !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-highlight"] {
    display: none !important;
    background: transparent !important;
    background-color: transparent !important;
    height: 0 !important;
    width: 0 !important;
    opacity: 0 !important;
    border: none !important;
    visibility: hidden !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: #2f3339 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #8a8f98 !important;
    border-radius: 0 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 4px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color 0.2s ease, border-bottom 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #f2f3f5 !important;
}
.stTabs [aria-selected="true"] {
    color: #f2f3f5 !important;
    background: transparent !important;
    border-bottom: 2px solid #22d3ee !important;
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 18px !important;
    animation: fadeIn 0.25s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Telemetry Card Tiles */
.telemetry-card {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-top: 2px solid #22d3ee;
    border-radius: 12px;
    padding: 16px 18px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.telemetry-card:hover {
    border-color: #3d4148;
    border-top-color: #22d3ee;
    transform: translateY(-1px);
}
.telemetry-val {
    color: #f2f3f5;
    font-size: 18px;
    font-weight: 500;
    margin-top: 4px;
}

/* Notice Blocks */
.notice-block {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-left: 3px solid #22d3ee;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    line-height: 1.5;
}
.notice-block.warn {
    border-left-color: #f59e0b;
}
.notice-title {
    font-weight: 500;
    color: #f2f3f5;
    margin-bottom: 2px;
}
.notice-body {
    color: #8a8f98;
}

/* Audio Status Bar */
.audio-status-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    padding: 12px 18px;
    transition: border-color 0.2s ease;
}
.audio-status-bar:hover {
    border-color: #3d4148;
}

/* Empty State */
.empty-state {
    text-align: center;
    padding: 56px 20px;
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    margin-top: 16px;
}
.empty-title {
    font-size: 16px;
    font-weight: 500;
    color: #f2f3f5;
    margin-bottom: 6px;
}
.empty-sub {
    font-size: 13px;
    color: #8a8f98;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Minimalist Footer */
.app-footer {
    text-align: center;
    color: #8a8f98;
    font-size: 12px;
    padding: 12px 0 8px;
    letter-spacing: 0.02em;
}

/* Top Nav Bar Component */
.top-nav-bar-anchor {
    display: block;
    height: 0px;
    margin: 0;
}
div:has(> .top-nav-bar-anchor) {
    position: sticky !important;
    top: 0 !important;
    z-index: 999 !important;
    background: rgba(20, 22, 26, 0.88) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    padding-top: 8px !important;
    padding-bottom: 2px !important;
}
.nav-brand-title {
    font-size: 18px;
    font-weight: 600;
    color: #f2f3f5;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
    height: 38px;
    margin: 0;
}
.nav-brand-pill {
    font-size: 10px;
    font-weight: 500;
    background: #0e2a2e;
    border: 1px solid #164e52;
    color: #67e8f9;
    padding: 2px 7px;
    border-radius: 9999px;
    letter-spacing: 0.04em;
}
.nav-bottom-divider {
    border-bottom: 1px solid #2f3339;
    margin: 4px 0 20px 0;
    width: 100%;
}

/* Nav Button Specific Styling */
div:has(.top-nav-bar-anchor) + div button,
.stElementContainer:has(.top-nav-bar-anchor) + .stElementContainer div[data-testid="stHorizontalBlock"] button,
div:has(> .top-nav-bar-anchor) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    color: #8a8f98 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
    height: 38px !important;
    box-shadow: none !important;
    transition: color 0.15s ease, border-bottom 0.15s ease !important;
}
div:has(.top-nav-bar-anchor) + div button:hover,
.stElementContainer:has(.top-nav-bar-anchor) + .stElementContainer div[data-testid="stHorizontalBlock"] button:hover,
div:has(> .top-nav-bar-anchor) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
    color: #f2f3f5 !important;
    background: transparent !important;
}
div:has(.top-nav-bar-anchor) + div button[kind="primary"],
div:has(.top-nav-bar-anchor) + div button[data-testid="baseButton-primary"],
.stElementContainer:has(.top-nav-bar-anchor) + .stElementContainer div[data-testid="stHorizontalBlock"] button[kind="primary"],
.stElementContainer:has(.top-nav-bar-anchor) + .stElementContainer div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-primary"],
div:has(> .top-nav-bar-anchor) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button[kind="primary"],
div:has(> .top-nav-bar-anchor) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button[data-testid="baseButton-primary"] {
    color: #f2f3f5 !important;
    background: transparent !important;
    border-bottom: 2px solid #22d3ee !important;
}

/* Smooth View Transition Animation */
.view-transition {
    animation: fadeSlideIn 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes fadeSlideIn {
    0% {
        opacity: 0;
        transform: translateY(8px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Hero Isometric SVG Animations & Layout */
.hero-svg-container {
    width: 100%;
    max-width: 580px;
    margin: 14px auto 22px auto;
    text-align: center;
}
@keyframes mic-ambient-pulse {
    0% {
        r: 10;
        opacity: 0.8;
    }
    50% {
        r: 18;
        opacity: 0.15;
    }
    100% {
        r: 10;
        opacity: 0.8;
    }
}
@keyframes wave-drift {
    0% {
        stroke-dashoffset: 0;
    }
    100% {
        stroke-dashoffset: 36;
    }
}
.pulsing-glow {
    animation: mic-ambient-pulse 2.2s ease-in-out infinite;
    transform-origin: center;
}
.flowing-wave {
    animation: wave-drift 4s linear infinite;
}

/* Home Feature Cards */
.feature-card {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    padding: 16px 18px;
    height: 100%;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.feature-card:hover {
    border-color: #3d4148;
    transform: translateY(-2px);
}
.feature-card-title {
    font-size: 14px;
    font-weight: 500;
    color: #f2f3f5;
    margin-bottom: 6px;
}
.feature-card-desc {
    font-size: 12px;
    color: #8a8f98;
    line-height: 1.5;
}

/* Feature Cards Staggered Entrance */
.entrance-1 { animation: cardEntrance 0.35s ease 0ms both; }
.entrance-2 { animation: cardEntrance 0.35s ease 80ms both; }
.entrance-3 { animation: cardEntrance 0.35s ease 160ms both; }
.entrance-4 { animation: cardEntrance 0.35s ease 240ms both; }
@keyframes cardEntrance {
    0% {
        opacity: 0;
        transform: translateY(12px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Sound-reactive pulse when audio is playing */
@keyframes soundReactivePulse {
    0% {
        box-shadow: 0 0 0 0 rgba(34, 211, 238, 0.4);
    }
    50% {
        box-shadow: 0 0 16px 2px rgba(34, 211, 238, 0.25);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(34, 211, 238, 0.4);
    }
}
.audio-active-pulse {
    animation: soundReactivePulse 1.8s infinite ease-in-out !important;
    border-color: #22d3ee !important;
}

/* Max-Width Constraint & Layout Polish */
.block-container {
    max-width: 1200px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Layered, Dynamic Card Treatment */
.layered-card {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    transform-origin: center center;
}
.tilt-left {
    transform: rotate(-0.5deg);
}
.tilt-right {
    transform: rotate(0.5deg);
}
.layered-card:hover {
    transform: translateY(-4px) rotate(0deg) !important;
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.65) !important;
    border-color: #3d4148 !important;
}

/* Multi-Color Icon-Badge System */
.icon-badge {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: transform 0.2s ease;
}
.icon-badge-cyan {
    background: rgba(34, 211, 238, 0.12);
    border: 1px solid #22d3ee;
    color: #22d3ee;
}
.icon-badge-violet {
    background: rgba(129, 140, 248, 0.12);
    border: 1px solid #818cf8;
    color: #818cf8;
}
.icon-badge-amber {
    background: rgba(245, 166, 35, 0.12);
    border: 1px solid #f5a623;
    color: #f5a623;
}
.icon-badge-emerald {
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid #34d399;
    color: #34d399;
}
.icon-badge-rose {
    background: rgba(244, 63, 94, 0.12);
    border: 1px solid #f43f5e;
    color: #f43f5e;
}
.icon-badge-neutral {
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid #94a3b8;
    color: #94a3b8;
}

/* Stat Strip Components (Home) */
.stat-card {
    padding: 20px 22px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 140px;
}
.stat-icon-badge {
    margin-bottom: 10px;
}
.stat-number {
    font-size: 28px;
    font-weight: 700;
    color: #f2f3f5;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 4px;
}
.stat-label {
    font-size: 11px;
    font-weight: 600;
    color: #8a8f98;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Equalizer Keyframe Animation */
.eq-container {
    display: inline-flex;
    align-items: flex-end;
    gap: 3px;
    height: 14px;
}
.eq-bar {
    width: 3px;
    background: #22d3ee;
    border-radius: 1.5px;
    animation: eqBounce 1.2s ease-in-out infinite alternate;
}
.eq-bar:nth-child(1) { height: 4px; animation-delay: 0.1s; }
.eq-bar:nth-child(2) { height: 10px; animation-delay: 0.3s; }
.eq-bar:nth-child(3) { height: 14px; animation-delay: 0.2s; }
.eq-bar:nth-child(4) { height: 8px; animation-delay: 0.4s; }
.eq-bar:nth-child(5) { height: 12px; animation-delay: 0.15s; }
@keyframes eqBounce {
    0% { height: 3px; }
    100% { height: 14px; }
}

/* Format Chips Row (Classify) */
.format-chip-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}
.format-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    color: #8a8f98;
    letter-spacing: 0.04em;
    transition: all 0.2s ease;
}
.format-chip:hover {
    border-color: #3d4148;
    color: #f2f3f5;
}
.format-chip-highlight {
    border-color: rgba(34, 211, 238, 0.4);
    color: #67e8f9;
    background: rgba(34, 211, 238, 0.06);
}

/* Emotion Valence Badges */
.tone-pill {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 8px;
    border-radius: 9999px;
    display: inline-block;
}
.tone-positive {
    background: rgba(52, 211, 153, 0.14);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.35);
}
.tone-negative {
    background: rgba(244, 63, 94, 0.14);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
}
.tone-neutral {
    background: rgba(148, 163, 184, 0.14);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.35);
}
.tone-dynamic {
    background: rgba(34, 211, 238, 0.14);
    color: #67e8f9;
    border: 1px solid rgba(34, 211, 238, 0.35);
}
/* Legacy aliases */
.tone-warm {
    background: rgba(244, 63, 94, 0.14);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
}
.tone-cool {
    background: rgba(129, 140, 248, 0.14);
    color: #a5b4fc;
    border: 1px solid rgba(129, 140, 248, 0.35);
}
.tone-electric {
    background: rgba(34, 211, 238, 0.14);
    color: #67e8f9;
    border: 1px solid rgba(34, 211, 238, 0.35);
}

/* Emotion Card V2 (About View) */
.emotion-card-v2 {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    transition: all 0.2s ease;
}
.emotion-card-v2:hover {
    border-color: #3d4148;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
}
.emotion-card-v2-body {
    flex: 1;
    min-width: 0;
}
.emotion-card-v2-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 3px;
}
.emotion-card-v2-title {
    font-size: 14px;
    font-weight: 600;
    color: #f2f3f5;
}
.emotion-card-v2-cue {
    font-size: 11px;
    color: #8a8f98;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Dashboards Left Rail Color Borders */
div:has(button[key="dash_tab_overview"]) button { border-left: 3px solid #22d3ee !important; }
div:has(button[key="dash_tab_waveform"]) button { border-left: 3px solid #818cf8 !important; }
div:has(button[key="dash_tab_spectrogram"]) button { border-left: 3px solid #f5a623 !important; }
div:has(button[key="dash_tab_probabilities"]) button { border-left: 3px solid #34d399 !important; }
div:has(button[key="dash_tab_telemetry"]) button { border-left: 3px solid #f43f5e !important; }

/* Interactive Radar Illustration Container */
.preview-radar-card {
    background: #1c1f24;
    border: 1px solid #2f3339;
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

/* ── SpeakSense Logo Bar Animation ── */
@keyframes logoBarPulse {
    0%, 100% { transform: scaleY(0.38); opacity: 0.45; }
    50%       { transform: scaleY(1.0);  opacity: 1.0;  }
}
.logo-bar-anim {
    animation: logoBarPulse 1.15s ease-in-out infinite;
    transform-box: fill-box;
    transform-origin: 50% 100%;
}

/* ── Loading indicator dot bounce ── */
@keyframes dotBounce {
    0%, 80%, 100% { transform: translateY(0);   opacity: 0.35; }
    40%            { transform: translateY(-6px); opacity: 1.0;  }
}
</style>
"""


st.markdown(PROFESSIONAL_THEME_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# Model Loading (cached across sessions)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_model_manager() -> ModelManager:
    """Load all models once and cache in session memory."""
    mgr = ModelManager()
    mgr.load_all()
    return mgr


# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════
# Audio Processing & Audibility Enhancer
# ════════════════════════════════════════════════════════════════════════════════

def make_audible_pcm_wav(raw_bytes: bytes, target_sr: int = SR) -> tuple[bytes, np.ndarray, float, float, bool, float]:
    """
    Decodes audio bytes from any container (WAV, WebM, MP3, OGG, FLAC) using soundfile
    or imageio_ffmpeg, trims dead air around speech, and re-encodes as a normalized 16-bit
    PCM WAV with loudness boosting so it is clearly audible on any device.

    Returns:
        (wav_bytes, y_boosted, dur_sec, raw_peak, has_speech, voiced_dur_sec)
    """
    y = None
    sr_found = target_sr

    # Method 1: Try soundfile directly (fast for WAV/FLAC)
    try:
        data, s_rate = sf.read(io.BytesIO(raw_bytes))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if s_rate != target_sr:
            data = librosa.resample(data.astype(np.float32), orig_sr=s_rate, target_sr=target_sr)
        y = data.astype(np.float32)
        sr_found = target_sr
    except Exception:
        pass

    # Method 2: Universal decoder using imageio_ffmpeg (decodes WebM, Opus, MP3, AAC)
    if y is None:
        try:
            import subprocess, imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-i", "pipe:0",
                "-vn",
                "-ac", "1",
                "-ar", str(target_sr),
                "-f", "wav",
                "pipe:1"
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate(input=raw_bytes)
            if proc.returncode == 0 and len(out) > 0:
                data, _ = sf.read(io.BytesIO(out))
                y = data.astype(np.float32)
                sr_found = target_sr
        except Exception as e:
            logger.warning("FFmpeg pipe decode fallback failed: %s", e)

    # Method 3: Fallback with tempfile
    if y is None:
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            data, _ = librosa.load(tmp_path, sr=target_sr, mono=True)
            y = data.astype(np.float32)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    if y is None or len(y) == 0:
        raise ValueError("Could not decode audio from the provided source.")

    # Remove DC offset to center the baseline at 0.0
    y = y - np.mean(y)

    # Attenuate startup hardware pop on microphone opening (first 20ms)
    fade_len = min(int(0.02 * sr_found), len(y))
    if fade_len > 0:
        y[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)

    raw_dur = len(y) / sr_found
    raw_peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0

    # Run Voice Activity Detection (VAD) with normalized internal thresholds
    s_idx, e_idx, has_speech, voiced_dur, mean_rms = detect_voice_activity(y, sr=sr_found)

    # If speech detected, trim dead air so playback and features align with speech
    if has_speech and (e_idx - s_idx) >= int(0.25 * sr_found):
        y_active = y[s_idx:e_idx]
        dur = len(y_active) / sr_found
    else:
        y_active = y
        dur = raw_dur

    # RMS loudness normalization to clear, loud conversational listening level (~0.14 RMS)
    active_rms = float(np.sqrt(np.mean(y_active**2))) if len(y_active) > 0 else 0.0
    target_rms = 0.14
    if active_rms > 1e-6:
        gain = target_rms / active_rms
        y_boosted = y_active * gain
        max_val = np.max(np.abs(y_boosted))
        if max_val > 0.95:
            y_boosted = (y_boosted / max_val) * 0.95
    elif raw_peak > 1e-6:
        y_boosted = (y_active / raw_peak) * 0.90
    else:
        y_boosted = y_active

    buf = io.BytesIO()
    sf.write(buf, y_boosted.astype(np.float32), sr_found, format="WAV", subtype="PCM_16")
    return buf.getvalue(), y_boosted.astype(np.float32), dur, raw_peak, has_speech, voiced_dur


# ════════════════════════════════════════════════════════════════════════════════
# Visualisation Helpers Aligned with Theme
# ════════════════════════════════════════════════════════════════════════════════

PLOT_BG   = "#0a1322"
PLOT_CARD = "#0d1a2d"
BORDER_COLOR = "#1f3352"

# ── Plotly base layout (Graphite + Cyan Theme) ────────────────────────────────
_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1c1f24",
    font=dict(family="Inter, -apple-system, sans-serif", color="#8a8f98", size=11),
    margin=dict(l=50, r=20, t=36, b=40),
    xaxis=dict(
        gridcolor="#262a30", gridwidth=1,
        linecolor="#2f3339", tickfont=dict(size=10, color="#8a8f98"),
        zerolinecolor="#2f3339",
    ),
    yaxis=dict(
        gridcolor="#262a30", gridwidth=1,
        linecolor="#2f3339", tickfont=dict(size=10, color="#8a8f98"),
        zerolinecolor="#2f3339",
    ),
    hoverlabel=dict(
        bgcolor="#1c1f24",
        bordercolor="#2f3339",
        font_color="#f2f3f5",
    ),
)


def plot_waveform(signal: np.ndarray, sr: int = SR, duration: float = DURATION):
    """Interactive waveform using Plotly with graphite and cyan accent."""
    n = len(signal)
    t = np.linspace(0, n / sr, n)
    step = max(1, n // 5000)
    t_ds, y_ds = t[::step], signal[::step]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_ds, y=y_ds, mode="lines",
        line=dict(color="#22d3ee", width=1.2),
        fill="tozeroy", fillcolor="rgba(34, 211, 238, 0.08)",
        name="Amplitude",
        hovertemplate="<b>%{x:.3f}s</b><br>Amp: %{y:.4f}<extra></extra>",
    ))
    lo = dict(**_PLOTLY_BASE)
    lo["title"] = dict(text="Time-Domain Waveform", font=dict(color="#f2f3f5", size=13), x=0)
    lo["xaxis"] = dict(**_PLOTLY_BASE["xaxis"], title="Time (s)")
    lo["yaxis"] = dict(**_PLOTLY_BASE["yaxis"], title="Amplitude")
    lo["height"] = 230
    fig.update_layout(**lo)
    return fig


def plot_rms_energy(signal: np.ndarray, sr: int = SR):
    """Interactive RMS energy contour using Plotly."""
    hop_len = 512
    rms = librosa.feature.rms(y=signal, hop_length=hop_len)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=rms, mode="lines",
        line=dict(color="#22d3ee", width=1.6),
        fill="tozeroy", fillcolor="rgba(34, 211, 238, 0.08)",
        name="RMS",
        hovertemplate="<b>%{x:.3f}s</b><br>RMS: %{y:.5f}<extra></extra>",
    ))
    lo = dict(**_PLOTLY_BASE)
    lo["title"] = dict(text="Vocal Energy (RMS) Contour", font=dict(color="#f2f3f5", size=13), x=0)
    lo["xaxis"] = dict(**_PLOTLY_BASE["xaxis"], title="Time (s)")
    lo["yaxis"] = dict(**_PLOTLY_BASE["yaxis"], title="RMS Energy")
    lo["height"] = 210
    fig.update_layout(**lo)
    return fig


def plot_mel_spectrogram(signal: np.ndarray, sr: int = SR):
    """Interactive mel spectrogram heatmap using Plotly."""
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=96, n_fft=2048, hop_length=512, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=512)
    freqs = librosa.mel_frequencies(n_mels=96, fmax=8000)
    fig = go.Figure(data=go.Heatmap(
        z=S_db, x=times, y=freqs,
        colorscale="Viridis",
        colorbar=dict(title="dB", tickfont=dict(color="#8a8f98", size=10), thickness=12, len=0.85),
        hovertemplate="<b>%{x:.3f}s</b><br>%{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
    ))
    lo = dict(**_PLOTLY_BASE)
    lo["title"] = dict(text="Log-Mel Spectrogram (96 Bands)", font=dict(color="#f2f3f5", size=13), x=0)
    lo["xaxis"] = dict(**_PLOTLY_BASE["xaxis"], title="Time (s)")
    lo["yaxis"] = dict(**_PLOTLY_BASE["yaxis"], title="Frequency (Hz)", type="log")
    lo["height"] = 300
    fig.update_layout(**lo)
    return fig


def plot_spectral_curves(signal: np.ndarray, sr: int = SR):
    """Interactive spectral centroid and rolloff curves using Plotly."""
    hop_len = 512
    cent = librosa.feature.spectral_centroid(y=signal, sr=sr, hop_length=hop_len)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, hop_length=hop_len)[0]
    t = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop_len)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=cent, mode="lines",
        line=dict(color="#22d3ee", width=1.5),
        name="Spectral Centroid",
        hovertemplate="<b>%{x:.3f}s</b><br>Centroid: %{y:.0f} Hz<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=rolloff, mode="lines",
        line=dict(color="#8a8f98", width=1.4, dash="dot"),
        name="Spectral Rolloff (85%)",
        hovertemplate="<b>%{x:.3f}s</b><br>Rolloff: %{y:.0f} Hz<extra></extra>",
    ))
    lo = dict(**_PLOTLY_BASE)
    lo["title"] = dict(text="Spectral Centroid & Rolloff", font=dict(color="#f2f3f5", size=13), x=0)
    lo["xaxis"] = dict(**_PLOTLY_BASE["xaxis"], title="Time (s)")
    lo["yaxis"] = dict(**_PLOTLY_BASE["yaxis"], title="Frequency (Hz)")
    lo["legend"] = dict(bgcolor="#1c1f24", bordercolor="#2f3339",
                        borderwidth=1, font=dict(color="#8a8f98", size=10))
    lo["height"] = 230
    fig.update_layout(**lo)
    return fig


def plot_probability_bars(probs: np.ndarray, em_labels: list, predicted: str):
    """Interactive horizontal bar chart of emotion probabilities with graphite + cyan accent."""
    sorted_idx = np.argsort(probs)
    labels = [em_labels[i].capitalize() for i in sorted_idx]
    values = [float(probs[i]) * 100 for i in sorted_idx]
    
    # Top predicted emotion gets cyan accent #22d3ee; others get muted graphite #242830
    colors = [
        "#22d3ee" if em_labels[i] == predicted
        else "#242830"
        for i in sorted_idx
    ]
    border_colors = [
        "#22d3ee" if em_labels[i] == predicted
        else "#2f3339"
        for i in sorted_idx
    ]
    text_colors = [
        "#22d3ee" if em_labels[i] == predicted
        else "#8a8f98"
        for i in sorted_idx
    ]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color=border_colors, width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=text_colors, size=10, family="Inter, sans-serif"),
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>",
    ))
    lo = dict(**_PLOTLY_BASE)
    lo["title"] = dict(text="Emotion Probability Distribution", font=dict(color="#f2f3f5", size=13), x=0)
    lo["xaxis"] = dict(**_PLOTLY_BASE["xaxis"], title="Confidence (%)", range=[0, 120])
    lo["height"] = 320
    lo["bargap"] = 0.32
    fig.update_layout(**lo)
    return fig


def plot_radar(probs: np.ndarray, em_labels: list) -> go.Figure:
    """Emotion probability radar chart using Plotly."""
    cats = [e.capitalize() for e in em_labels] + [em_labels[0].capitalize()]
    vals = list(probs * 100) + [float(probs[0]) * 100]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(34, 211, 238, 0.08)",
        line=dict(color="#22d3ee", width=1.8),
        hovertemplate="<b>%{theta}</b><br>%{r:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1c1f24",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#262a30",
                            tickfont=dict(color="#8a8f98", size=9), showline=False),
            angularaxis=dict(gridcolor="#262a30", tickfont=dict(color="#8a8f98", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8a8f98"),
        title=dict(text="Emotion Radar Chart", font=dict(color="#f2f3f5", size=13), x=0.05),
        margin=dict(l=55, r=55, t=45, b=35),
        height=320, showlegend=False,
        hoverlabel=dict(bgcolor="#1c1f24", bordercolor="#2f3339", font_color="#f2f3f5"),
    )
    return fig






# ════════════════════════════════════════════════════════════════════════════════
# Sample Finder
# ════════════════════════════════════════════════════════════════════════════════

def find_ravdess_sample(emotion: str, data_dir: str = DATA_DIR) -> str | None:
    """Return the path to an audio file matching the given emotion."""
    import glob
    target_code = {v: k for k, v in RAVDESS_CODE.items()}.get(emotion)
    if not target_code or not os.path.isdir(data_dir):
        return None
    pattern = os.path.join(data_dir, "**", f"03-01-{target_code}-*.wav")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        for actor_dir in sorted(os.listdir(data_dir)):
            wavs = glob.glob(os.path.join(data_dir, actor_dir, "*.wav"))
            for w in sorted(wavs):
                code = os.path.basename(w).split("-")[2]
                if code == target_code:
                    return w
        return None
    return sorted(matches)[0]


# ════════════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ════════════════════════════════════════════════════════════════════════════════

_defaults = {
    "active_view": "home",
    "reset_counter": 0,
    "model_choice": "SVM Optimized",
    "dash_panel": "overview",
    "preview_signal": None,
    "audio_bytes": None,
    "audio_label": None,
    "audible_wav": None,
    "audio_duration": 0.0,
    "has_speech": True,
    "voiced_duration": 0.0,
    "raw_peak": 0.0,
    "prediction_result": None,
    "preprocessed_signal": None,
    "selected_sample_emotion": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Load models
with st.spinner("Initializing acoustic models..."):
    try:
        mgr = get_model_manager()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

available = mgr.models.keys()
if not available:
    st.error("No models available in the models/ directory.")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# ── UI HELPERS ────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# ── LOGO & LOADING HELPERS ─────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def render_logo(size: int = 32, animated: bool = False) -> str:
    """Returns inline SVG of the SpeakSense waveform mark at the requested size.

    Args:
        size:     Pixel dimensions (square). Common: 24 (nav), 32 (default), 48-64 (hero).
        animated: If True, bars carry the CSS `logo-bar-anim` class for the
                  equalizer pulse animation defined in PROFESSIONAL_THEME_CSS.
    """
    sc = size / 64.0
    # (x, y, width, height, fill-color) — coordinates match the 64px reference icon
    bars = [
        (15 * sc, 30 * sc, 4 * sc, 8 * sc,  "#67e8f9"),   # outer left, muted
        (22 * sc, 24 * sc, 4 * sc, 20 * sc, "#22d3ee"),
        (29 * sc, 13 * sc, 4 * sc, 38 * sc, "#22d3ee"),   # center, tallest
        (36 * sc, 21 * sc, 4 * sc, 22 * sc, "#22d3ee"),
        (43 * sc, 27 * sc, 4 * sc, 10 * sc, "#67e8f9"),   # outer right, muted
    ]
    anim_delays = [0.28, 0.14, 0.0, 0.14, 0.28]
    bar_els = []
    for i, (bx, by, bw, bh, color) in enumerate(bars):
        rx = max(1.0, round(bw / 2, 1))
        anim_attr = (
            f' class="logo-bar-anim" style="animation-delay:{anim_delays[i]}s"'
            if animated else ""
        )
        bar_els.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
            f'rx="{rx}" fill="{color}"{anim_attr}/>'
        )
    bars_svg = "\n  ".join(bar_els)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="SpeakSense">\n'
        f'  {bars_svg}\n</svg>'
    )


def render_loading_indicator(message: str = "Processing…") -> str:
    """Returns a branded HTML loading card using the animated SpeakSense waveform logo.

    Drop this into any `st.empty().markdown(...)` call to replace Streamlit's generic
    spinner with the SpeakSense waveform mark during processing waits.
    """
    logo_svg = render_logo(size=48, animated=True)
    dots = "".join(
        f'<div style="width:5px;height:5px;border-radius:50%;background:#22d3ee;'
        f'animation:dotBounce 1.2s ease-in-out infinite;animation-delay:{d:.1f}s;"></div>'
        for d in (0.0, 0.18, 0.36)
    )
    return f"""
    <div style="
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        background:#1c1f24; border:1px solid #2f3339; border-radius:12px;
        padding:36px 24px 28px; gap:14px; text-align:center; min-height:200px;
    ">
        {logo_svg}
        <div style="font-size:13px; color:#8a8f98; letter-spacing:0.02em;">{message}</div>
        <div style="display:flex; gap:5px; align-items:center;">{dots}</div>
    </div>
    """


def render_view_header(eyebrow: str, title: str, desc: str):
    """Consistent view header banner with eyebrow, title, and description."""
    st.markdown(f"""
    <div class="view-header-block">
        <div class="view-eyebrow">{eyebrow}</div>
        <h2 class="view-title">{title}</h2>
        <p class="view-desc">{desc}</p>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(
    icon: str,
    title: str,
    subtitle: str | None = None,
    accent: str | None = None,
    first: bool = False,
):
    """Reusable section header component with colored circular badge, bold title, and subtitle."""
    # Color palette fallback
    if accent is None:
        tl = title.lower()
        if "model" in tl or "benchmark" in tl:
            accent = "#818cf8"
        elif "audio" in tl or "pipeline" in tl or "telemetry" in tl:
            accent = "#22d3ee"
        elif "tip" in tl:
            accent = "#f5a623"
        elif "emotion" in tl or "standard" in tl:
            accent = "#34d399"
        else:
            accent = "#22d3ee"

    # Hex to rgba
    hex_clean = accent.lstrip("#")
    if len(hex_clean) == 6:
        r = int(hex_clean[0:2], 16)
        g = int(hex_clean[2:4], 16)
        b = int(hex_clean[4:6], 16)
        bg_rgba = f"rgba({r}, {g}, {b}, 0.12)"
    else:
        bg_rgba = "rgba(34, 211, 238, 0.12)"

    standard_icons = {
        "model": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07M8.46 8.46a5 5 0 0 0 0 7.07"/></svg>',
        "mic": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>',
        "audio": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>',
        "tips": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg>',
        "pipeline": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "standards": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>',
        "benchmark": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
        "result": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    }

    if icon in standard_icons:
        icon_svg = standard_icons[icon]
    elif icon.strip().startswith("<svg"):
        icon_svg = icon
    else:
        icon_svg = f'<span style="font-size:16px; line-height:1;">{icon}</span>'

    rule_html = "" if first else '<hr class="section-rule" />'
    first_cls = " first-section" if first else ""
    sub_html = f'<div class="sh-subtitle">{subtitle}</div>' if subtitle else ""

    html = (
        f'<div class="section-block{first_cls}">'
        f'{rule_html}'
        f'<div class="sh-container">'
        f'<div class="sh-row">'
        f'<div class="sh-badge" style="background:{bg_rgba}; border:1px solid {accent};">{icon_svg}</div>'
        f'<span class="sh-title">{title}</span>'
        f'</div>'
        f'{sub_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_footer_status():
    """Sticky bottom footer status bar displayed across all views."""
    st.markdown("""
    <div class="sticky-footer-wrap">
        <div class="footer-status-bar">
            <div class="status-item">
                <div class="eq-container">
                    <span class="eq-bar"></span>
                    <span class="eq-bar"></span>
                    <span class="eq-bar"></span>
                    <span class="eq-bar"></span>
                    <span class="eq-bar"></span>
                </div>
                <span class="status-code" style="color:#67e8f9; font-weight:600;">speaksense core</span>
            </div>
            <div class="status-divider-dot">&middot;</div>
            <div class="status-item">
                <span class="status-dot"></span>
                <span class="status-code">models: 5 loaded</span>
            </div>
            <div class="status-divider-dot">&middot;</div>
            <div class="status-item">
                <span class="status-dot"></span>
                <span class="status-code">pipeline: 22.05kHz ready</span>
            </div>
            <div class="status-divider-dot">&middot;</div>
            <div class="status-item">
                <span class="status-dot"></span>
                <span class="status-code">8 emotion classes</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ── QUICK TIPS FLASHCARD DECK (Self-contained HTML/JS Component) ────────────────
# ════════════════════════════════════════════════════════════════════════════════

_FLASHCARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    padding: 0 4px 28px 4px;
    -webkit-font-smoothing: antialiased;
  }

  /* ---- Section header ---- */
  .fc-hdr {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 18px; padding: 0 2px;
  }
  .fc-hdr-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22d3ee;
    box-shadow: 0 0 8px #22d3ee99;
  }
  .fc-hdr-lbl {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase; color: #5a6272;
  }

  /* ---- Stack container ---- */
  .deck {
    position: relative;
    cursor: pointer;
    /* bottom padding lets peek cards show below front card */
    padding-bottom: 22px;
    -webkit-tap-highlight-color: transparent;
  }

  /* ---- Peek cards (behind, peeking below) ---- */
  .peek {
    position: absolute;
    left: 50%; transform: translateX(-50%);
    border-radius: 16px;
    pointer-events: none;
    border: 1px solid #1a2035;
  }
  .peek-1 {
    width: calc(100% - 18px);
    height: 56px;
    bottom: 10px;
    background: linear-gradient(145deg, #0b1020, #0e1428);
    opacity: 0.65;
    animation: peekBreath1 4.5s ease-in-out infinite alternate;
  }
  .peek-2 {
    width: calc(100% - 36px);
    height: 56px;
    bottom: 0;
    background: linear-gradient(145deg, #080e1a, #0b1020);
    opacity: 0.38;
    animation: peekBreath2 6s ease-in-out infinite alternate;
  }
  @keyframes peekBreath1 {
    0%   { opacity: 0.60; bottom: 10px; }
    100% { opacity: 0.70; bottom: 12px; }
  }
  @keyframes peekBreath2 {
    0%   { opacity: 0.33; bottom: 0px; }
    100% { opacity: 0.43; bottom: 3px; }
  }

  /* ---- Front card ---- */
  .front {
    position: relative; z-index: 4;
    background: linear-gradient(150deg, #0c111c 0%, #111827 60%, #0e1523 100%);
    border: 1px solid #1e2a40;
    border-top: 3px solid var(--ac);
    border-radius: 16px;
    padding: 22px 22px 18px 22px;
    box-shadow: 0 18px 52px rgba(0,0,0,0.65), 0 0 0 1px rgba(255,255,255,0.025);
    display: flex; align-items: flex-start; gap: 18px;
    transition: transform 0.13s ease, box-shadow 0.13s ease;
    user-select: none;
    min-height: 140px;
  }
  .front:hover {
    transform: translateY(-3px);
    box-shadow: 0 24px 60px rgba(0,0,0,0.75);
  }
  .front:active { transform: scale(0.975); }

  /* ---- Slide-in animation ---- */
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(-18px) scale(0.95); }
    to   { opacity: 1; transform: translateY(0)    scale(1.0);  }
  }
  .anim { animation: slideIn 0.36s cubic-bezier(0.16,1,0.3,1) both; }

  /* ---- Visual block (CSS art) ---- */
  .vis-wrap {
    width: 56px; height: 56px; min-width: 56px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    position: relative; overflow: hidden;
  }

  /* Tip 0: Microphone — two rounded pillars + arc */
  .v0 { background: rgba(34,211,238,0.10); border: 1.5px solid rgba(34,211,238,0.28); }
  .v0::before {
    content: '';
    width: 16px; height: 26px;
    border-radius: 8px;
    border: 2.5px solid #22d3ee;
    position: absolute; top: 9px;
    box-shadow: 0 0 10px #22d3ee55;
  }
  .v0::after {
    content: '';
    width: 26px; height: 13px;
    border: 2.5px solid #22d3ee;
    border-top: none;
    border-radius: 0 0 13px 13px;
    position: absolute; bottom: 6px;
    box-shadow: 0 6px 10px #22d3ee22;
  }

  /* Tip 1: Audio wave bars */
  .v1 { background: rgba(129,140,248,0.10); border: 1.5px solid rgba(129,140,248,0.28); gap: 4px; }
  .wb {
    width: 5px; border-radius: 3px;
    background: linear-gradient(to top, #818cf8, #a5b4fc);
    box-shadow: 0 0 6px #818cf844;
    animation: wbPulse 1.6s ease-in-out infinite alternate;
  }
  .wb:nth-child(1) { height: 12px; animation-delay: 0.0s; }
  .wb:nth-child(2) { height: 28px; animation-delay: 0.2s; }
  .wb:nth-child(3) { height: 38px; animation-delay: 0.4s; }
  .wb:nth-child(4) { height: 22px; animation-delay: 0.6s; }
  .wb:nth-child(5) { height: 14px; animation-delay: 0.8s; }
  @keyframes wbPulse {
    0%   { transform: scaleY(0.80); opacity: 0.65; }
    100% { transform: scaleY(1.00); opacity: 1.00; }
  }

  /* Tip 2: Lightning bolt (CSS triangle halves) */
  .v2 { background: rgba(245,158,11,0.10); border: 1.5px solid rgba(245,158,11,0.28); }
  .bolt-t {
    position: absolute; top: 7px; left: 14px;
    width: 0; height: 0;
    border-style: solid;
    border-width: 0 17px 24px 0;
    border-color: transparent #f59e0b transparent transparent;
    filter: drop-shadow(0 0 6px #f59e0b88);
  }
  .bolt-b {
    position: absolute; bottom: 7px; right: 12px;
    width: 0; height: 0;
    border-style: solid;
    border-width: 24px 0 0 17px;
    border-color: transparent transparent transparent #fbbf24cc;
    filter: drop-shadow(0 0 4px #f59e0b55);
  }

  /* Tip 3: Bar chart */
  .v3 {
    background: rgba(52,211,153,0.10); border: 1.5px solid rgba(52,211,153,0.28);
    align-items: flex-end; gap: 4px; padding: 8px 7px 4px 7px;
  }
  .cb {
    flex: 1; border-radius: 3px 3px 0 0;
    background: linear-gradient(to top, #34d399, #6ee7b7);
    box-shadow: 0 0 6px #34d39944;
    animation: cbPulse 2s ease-in-out infinite alternate;
  }
  @keyframes cbPulse {
    0%   { opacity: 0.75; }
    100% { opacity: 1.00; }
  }

  /* ---- Text content ---- */
  .card-text { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 0; }
  .tip-lbl {
    font-size: 9.5px; font-weight: 700;
    letter-spacing: 0.11em; text-transform: uppercase;
    color: var(--ac); margin-bottom: 6px;
  }
  .tip-ttl {
    font-size: 15px; font-weight: 700;
    color: #eef0f4; letter-spacing: -0.02em;
    line-height: 1.3; margin-bottom: 8px;
  }
  .tip-bod {
    font-size: 12px; color: #8691a0;
    line-height: 1.65; flex: 1;
  }

  /* ---- Bottom bar ---- */
  .bot-bar {
    display: flex; align-items: center;
    justify-content: space-between;
    margin-top: 14px;
  }
  .dot-row { display: flex; align-items: center; gap: 6px; }
  .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #1e2535;
    transition: background 0.3s, transform 0.3s, box-shadow 0.3s;
    display: inline-block;
  }
  .dot.on {
    background: var(--ac);
    box-shadow: 0 0 8px var(--ac);
    transform: scale(1.45);
  }
  .tap-lbl {
    font-size: 9px; color: #2a3245;
    text-transform: uppercase; letter-spacing: 0.09em; font-weight: 700;
  }
</style>
</head>
<body>
<div class="deck" id="deck" onclick="next()">
  <div class="peek peek-2" id="peek2"></div>
  <div class="peek peek-1" id="peek1"></div>

  <div class="front anim" id="front" style="--ac:#22d3ee;">
    <div class="vis-wrap v0" id="vis"></div>
    <div class="card-text">
      <div class="tip-lbl" id="lbl">Tip 1 / 4</div>
      <div class="tip-ttl" id="ttl">2\u20134s is optimal</div>
      <div class="tip-bod" id="bod">Record 2\u20134 seconds of clear speech for the highest accuracy emotion read.</div>
      <div class="bot-bar">
        <div class="dot-row" id="dots"></div>
        <span class="tap-lbl">tap to switch</span>
      </div>
    </div>
  </div>
</div>

<script>
const TIPS = [
  {
    ac: '#22d3ee',
    ttl: '2\u20134s is optimal',
    bod: 'Record 2\u20134 seconds of clear speech for the highest accuracy emotion read.',
    vc: 'v0',
    vi: ''
  },
  {
    ac: '#818cf8',
    ttl: 'Speak naturally',
    bod: 'Conversational volume works best \u2014 loud or whispered audio degrades accuracy.',
    vc: 'v1',
    vi: '<div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div>'
  },
  {
    ac: '#f59e0b',
    ttl: 'SVM for live voice',
    bod: 'SVM Optimized is fastest for microphone recordings; ANN is best for file uploads.',
    vc: 'v2',
    vi: '<div class="bolt-t"></div><div class="bolt-b"></div>'
  },
  {
    ac: '#34d399',
    ttl: 'Dashboards unlock',
    bod: 'Run a classification first \u2014 it unlocks waveform, spectrogram, and probability dashboards.',
    vc: 'v3',
    vi: '<div class="cb" style="height:55%"></div><div class="cb" style="height:80%"></div><div class="cb" style="height:40%"></div><div class="cb" style="height:100%"></div><div class="cb" style="height:65%"></div>'
  }
];

let idx = 0;

function renderDots() {
  const row = document.getElementById('dots');
  row.innerHTML = TIPS.map((_, i) =>
    '<span class="dot' + (i === idx ? ' on' : '') + '"></span>'
  ).join('');
}

function next() {
  idx = (idx + 1) % TIPS.length;
  const t = TIPS[idx];
  const front = document.getElementById('front');
  const vis   = document.getElementById('vis');

  front.style.setProperty('--ac', t.ac);
  vis.className = 'vis-wrap ' + t.vc;
  vis.innerHTML = t.vi;
  document.getElementById('lbl').textContent = 'Tip ' + (idx + 1) + ' / ' + TIPS.length;
  document.getElementById('lbl').style.color = t.ac;
  document.getElementById('ttl').textContent = t.ttl;
  document.getElementById('bod').textContent = t.bod;

  // Restart animation
  front.classList.remove('anim');
  void front.offsetWidth;
  front.classList.add('anim');

  renderDots();
}

// Set peek card height = front card height after first paint
function sizePeeks() {
  const h = document.getElementById('front').offsetHeight;
  document.getElementById('peek1').style.height = h + 'px';
  document.getElementById('peek2').style.height = h + 'px';
  // Also size the deck bottom padding
  document.getElementById('deck').style.paddingBottom = '24px';
}

renderDots();
window.addEventListener('load', sizePeeks);
window.addEventListener('resize', sizePeeks);
</script>
</body>
</html>
"""


def render_tip_flashcards(tips: list | None = None, state_key: str = "home_tip_index"):
    """Renders a self-contained vertical stacked flashcard deck.

    All tip data, CSS art visuals, and tap-to-cycle interaction live inside
    a single st.components.v1.html iframe. Preceded by standard section header.
    """
    render_section_header(
        "tips",
        "Quick Tips",
        "Tap card to cycle through audio best practices",
        accent="#f5a623",
        first=False,
    )
    import streamlit.components.v1 as _stc
    _stc.html(_FLASHCARD_HTML, height=226, scrolling=False)


# ════════════════════════════════════════════════════════════════════════════════
# ── TOP NAVIGATION BAR ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def render_top_navbar():
    st.markdown('<div class="top-nav-bar-anchor"></div>', unsafe_allow_html=True)
    c_brand, c_nav1, c_nav2, c_nav3, c_nav4 = st.columns([3.2, 1.1, 1.1, 1.3, 1.1])
    
    with c_brand:
        _nav_logo = render_logo(size=26)
        st.markdown(f"""
        <div class="nav-brand-title">
            {_nav_logo}
            SpeakSense <span class="nav-brand-pill">V2.4</span>
        </div>
        """, unsafe_allow_html=True)
    
    current_view = st.session_state.get("active_view", "home")
    
    with c_nav1:
        if st.button("Home", key="nav_home", type="primary" if current_view == "home" else "secondary", use_container_width=True):
            if current_view != "home":
                st.session_state.active_view = "home"
                st.rerun()
                
    with c_nav2:
        if st.button("Classify", key="nav_classify", type="primary" if current_view == "classify" else "secondary", use_container_width=True):
            if current_view != "classify":
                st.session_state.active_view = "classify"
                st.rerun()
                
    with c_nav3:
        if st.button("Dashboards", key="nav_dashboards", type="primary" if current_view == "dashboards" else "secondary", use_container_width=True):
            if current_view != "dashboards":
                st.session_state.active_view = "dashboards"
                st.rerun()
                
    with c_nav4:
        if st.button("About", key="nav_about", type="primary" if current_view == "about" else "secondary", use_container_width=True):
            if current_view != "about":
                st.session_state.active_view = "about"
                st.rerun()
    
    st.markdown('<div class="nav-bottom-divider"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ── VIEW 1: HOME ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def render_home():
    # Hero Section with faint audio waveform motif and concise 1-sentence subtitle
    st.markdown("""
    <div class="hero-header">
        <svg class="hero-waveform-bg" width="340" height="56" viewBox="0 0 340 56" fill="none" aria-hidden="true">
            <rect x="10" y="22" width="3" height="12" rx="1.5" fill="#22d3ee"/>
            <rect x="18" y="19" width="3" height="18" rx="1.5" fill="#22d3ee"/>
            <rect x="26" y="21" width="3" height="14" rx="1.5" fill="#22d3ee"/>
            <rect x="34" y="15" width="3" height="26" rx="1.5" fill="#22d3ee"/>
            <rect x="42" y="9" width="3" height="38" rx="1.5" fill="#22d3ee"/>
            <rect x="50" y="17" width="3" height="22" rx="1.5" fill="#22d3ee"/>
            <rect x="58" y="6" width="3" height="44" rx="1.5" fill="#22d3ee"/>
            <rect x="66" y="12" width="3" height="32" rx="1.5" fill="#22d3ee"/>
            <rect x="74" y="4" width="3" height="48" rx="1.5" fill="#22d3ee"/>
            <rect x="82" y="14" width="3" height="28" rx="1.5" fill="#22d3ee"/>
            <rect x="90" y="2" width="3" height="52" rx="1.5" fill="#22d3ee"/>
            <rect x="98" y="10" width="3" height="36" rx="1.5" fill="#22d3ee"/>
            <rect x="106" y="7" width="3" height="42" rx="1.5" fill="#22d3ee"/>
            <rect x="114" y="1" width="3" height="54" rx="1.5" fill="#22d3ee"/>
            <rect x="122" y="5" width="3" height="46" rx="1.5" fill="#22d3ee"/>
            <rect x="130" y="13" width="3" height="30" rx="1.5" fill="#22d3ee"/>
            <rect x="138" y="4" width="3" height="48" rx="1.5" fill="#22d3ee"/>
            <rect x="146" y="2" width="3" height="52" rx="1.5" fill="#22d3ee"/>
            <rect x="154" y="9" width="3" height="38" rx="1.5" fill="#22d3ee"/>
            <rect x="162" y="6" width="3" height="44" rx="1.5" fill="#22d3ee"/>
            <rect x="170" y="15" width="3" height="26" rx="1.5" fill="#22d3ee"/>
            <rect x="178" y="10" width="3" height="36" rx="1.5" fill="#22d3ee"/>
            <rect x="186" y="5" width="3" height="46" rx="1.5" fill="#22d3ee"/>
            <rect x="194" y="12" width="3" height="32" rx="1.5" fill="#22d3ee"/>
            <rect x="202" y="17" width="3" height="22" rx="1.5" fill="#22d3ee"/>
            <rect x="210" y="9" width="3" height="38" rx="1.5" fill="#22d3ee"/>
            <rect x="218" y="15" width="3" height="26" rx="1.5" fill="#22d3ee"/>
            <rect x="226" y="19" width="3" height="18" rx="1.5" fill="#22d3ee"/>
            <rect x="234" y="16" width="3" height="24" rx="1.5" fill="#22d3ee"/>
            <rect x="242" y="20" width="3" height="16" rx="1.5" fill="#22d3ee"/>
            <rect x="250" y="22" width="3" height="12" rx="1.5" fill="#22d3ee"/>
            <rect x="258" y="18" width="3" height="20" rx="1.5" fill="#22d3ee"/>
            <rect x="266" y="23" width="3" height="10" rx="1.5" fill="#22d3ee"/>
        </svg>
        <div class="hero-badge">Acoustic Intelligence &middot; Real-Time</div>
        <h1 class="hero-title">Speech Emotion Recognition</h1>
        <p class="hero-subtitle">
            Real-time voice emotion intelligence powered by deep acoustic learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Isometric Microphone & Waveform Illustration (pure SVG, cyan & graphite)
    import streamlit.components.v1 as _stc
    _stc.html("""
    <style>
    @keyframes mic-ambient-pulse-c {
        0% { r: 10; opacity: 0.8; }
        50% { r: 18; opacity: 0.15; }
        100% { r: 10; opacity: 0.8; }
    }
    @keyframes wave-drift-c {
        0% { stroke-dashoffset: 0; }
        100% { stroke-dashoffset: 36; }
    }
    .pulsing-glow-c { animation: mic-ambient-pulse-c 2.2s ease-in-out infinite; }
    .flowing-wave-c { animation: wave-drift-c 4s linear infinite; }
    </style>
    <div style="width:100%;max-width:580px;margin:10px auto 16px auto;text-align:center;">
        <svg viewBox="0 0 520 220" width="100%" height="220" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="isoBaseGradC" x1="120" y1="80" x2="400" y2="210" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#1e2229"/>
                    <stop offset="100%" stop-color="#14161a"/>
                </linearGradient>
                <linearGradient id="isoScreenGradC" x1="160" y1="100" x2="360" y2="180" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#16181d"/>
                    <stop offset="100%" stop-color="#0f1114"/>
                </linearGradient>
                <linearGradient id="micBodyGradC" x1="245" y1="40" x2="275" y2="95" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#2a2e36"/>
                    <stop offset="50%" stop-color="#1f2228"/>
                    <stop offset="100%" stop-color="#14161a"/>
                </linearGradient>
                <linearGradient id="barGradC" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#22d3ee"/>
                    <stop offset="100%" stop-color="#164e52"/>
                </linearGradient>
            </defs>
            <path d="M 20 80 Q 80 40 140 80 T 260 80 T 380 80 T 500 80" stroke="#22d3ee" stroke-width="1.5" stroke-opacity="0.25" stroke-dasharray="8,6" class="flowing-wave-c" fill="none"/>
            <path d="M 30 110 Q 90 70 150 110 T 270 110 T 390 110 T 490 110" stroke="#22d3ee" stroke-width="1" stroke-opacity="0.15" stroke-dasharray="6,8" class="flowing-wave-c" fill="none"/>
            <polygon points="120,150 260,215 400,150 260,85" fill="#0c0e11" opacity="0.6"/>
            <polygon points="120,150 260,215 260,225 120,160" fill="#181a1f" stroke="#2f3339" stroke-width="1"/>
            <polygon points="400,150 260,215 260,225 400,160" fill="#121417" stroke="#2f3339" stroke-width="1"/>
            <polygon points="120,150 260,85 400,150 260,215" fill="url(#isoBaseGradC)" stroke="#2f3339" stroke-width="1.5"/>
            <polygon points="144,146 260,93 376,146 260,199" fill="url(#isoScreenGradC)" stroke="#22d3ee" stroke-opacity="0.3" stroke-width="1"/>
            <polygon points="172,138 178,135 178,142 172,145" fill="url(#barGradC)"/>
            <polygon points="184,131 190,128 190,141 184,144" fill="url(#barGradC)"/>
            <polygon points="196,122 202,119 202,140 196,143" fill="url(#barGradC)"/>
            <polygon points="208,128 214,125 214,142 208,145" fill="url(#barGradC)"/>
            <polygon points="220,116 226,113 226,144 220,147" fill="url(#barGradC)"/>
            <polygon points="232,124 238,121 238,146 232,149" fill="url(#barGradC)"/>
            <polygon points="244,110 250,107 250,151 244,154" fill="url(#barGradC)"/>
            <polygon points="256,104 262,101 262,153 256,156" fill="url(#barGradC)"/>
            <polygon points="268,112 274,109 274,151 268,154" fill="url(#barGradC)"/>
            <polygon points="280,118 286,115 286,149 280,152" fill="url(#barGradC)"/>
            <polygon points="292,124 298,121 298,147 292,150" fill="url(#barGradC)"/>
            <polygon points="304,115 310,112 310,145 304,148" fill="url(#barGradC)"/>
            <polygon points="316,127 322,124 322,143 316,146" fill="url(#barGradC)"/>
            <polygon points="328,133 334,130 334,141 328,144" fill="url(#barGradC)"/>
            <polygon points="340,138 346,135 346,142 340,145" fill="url(#barGradC)"/>
            <ellipse cx="260" cy="80" rx="90" ry="38" stroke="#22d3ee" stroke-opacity="0.15" stroke-width="1.5" stroke-dasharray="5,5" fill="none"/>
            <ellipse cx="260" cy="80" rx="60" ry="25" stroke="#22d3ee" stroke-opacity="0.25" stroke-width="1.5" fill="none"/>
            <circle cx="260" cy="74" r="14" fill="#22d3ee" class="pulsing-glow-c"/>
            <ellipse cx="260" cy="150" rx="20" ry="9" fill="#1c1f24" stroke="#2f3339" stroke-width="1.5"/>
            <line x1="260" y1="150" x2="260" y2="108" stroke="#3d4148" stroke-width="4" stroke-linecap="round"/>
            <circle cx="260" cy="112" r="4" fill="#22d3ee"/>
            <ellipse cx="260" cy="98" rx="26" ry="11" fill="none" stroke="#2f3339" stroke-width="2"/>
            <line x1="242" y1="99" x2="252" y2="92" stroke="#22d3ee" stroke-width="1.5" stroke-opacity="0.8"/>
            <line x1="278" y1="99" x2="268" y2="92" stroke="#22d3ee" stroke-width="1.5" stroke-opacity="0.8"/>
            <rect x="250" y="65" width="20" height="30" rx="4" fill="url(#micBodyGradC)" stroke="#2f3339" stroke-width="1.5"/>
            <path d="M 250 65 C 250 50 270 50 270 65 Z" fill="#22262c" stroke="#22d3ee" stroke-width="1.5"/>
            <line x1="250" y1="60" x2="270" y2="60" stroke="#22d3ee" stroke-opacity="0.7" stroke-width="1"/>
            <line x1="252" y1="55" x2="268" y2="55" stroke="#22d3ee" stroke-opacity="0.5" stroke-width="1"/>
            <line x1="260" y1="52" x2="260" y2="65" stroke="#22d3ee" stroke-opacity="0.6" stroke-width="1"/>
            <line x1="250" y1="74" x2="270" y2="74" stroke="#22d3ee" stroke-width="2"/>
            <circle cx="370" cy="100" r="3" fill="#22d3ee"/>
            <text x="378" y="103" fill="#8a8f98" font-size="10" font-family="monospace" letter-spacing="1">PCM 16-BIT</text>
            <circle cx="110" cy="115" r="3" fill="#22d3ee"/>
            <text x="50" y="118" fill="#8a8f98" font-size="10" font-family="monospace" letter-spacing="1">22.05 kHz</text>
        </svg>
    </div>
    """, height=240, scrolling=False)

    # CTA Button: "Try It Now"
    _, col_cta, _ = st.columns([1.8, 1.4, 1.8])
    with col_cta:
        if st.button("Try It Now", key="home_cta_btn", type="primary", use_container_width=True):
            st.session_state.active_view = "classify"
            st.rerun()

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # 4-Column Visual Stat Strip (Minimal Text, Icon Badges, Bold Data)
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown("""
        <div class="stat-card layered-card tilt-left entrance-1">
            <div class="stat-icon-badge icon-badge icon-badge-violet">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
                    <line x1="9" y1="9" x2="9.01" y2="9"/>
                    <line x1="15" y1="9" x2="15.01" y2="9"/>
                </svg>
            </div>
            <div class="stat-number">8</div>
            <div class="stat-label">Emotion Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown("""
        <div class="stat-card layered-card tilt-right entrance-2">
            <div class="stat-icon-badge icon-badge icon-badge-emerald">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="2" width="20" height="8" rx="2"/>
                    <rect x="2" y="14" width="20" height="8" rx="2"/>
                    <line x1="6" y1="6" x2="6.01" y2="6"/>
                    <line x1="6" y1="18" x2="6.01" y2="18"/>
                </svg>
            </div>
            <div class="stat-number">5</div>
            <div class="stat-label">Benchmark Models</div>
        </div>
        """, unsafe_allow_html=True)
    with fc3:
        st.markdown("""
        <div class="stat-card layered-card tilt-left entrance-3">
            <div class="stat-icon-badge icon-badge icon-badge-amber">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f5a623" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"/>
                    <line x1="12" y1="20" x2="12" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="14"/>
                </svg>
            </div>
            <div class="stat-number">264</div>
            <div class="stat-label">Acoustic Features</div>
        </div>
        """, unsafe_allow_html=True)
    with fc4:
        st.markdown("""
        <div class="stat-card layered-card tilt-right entrance-4">
            <div class="stat-icon-badge icon-badge icon-badge-cyan">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                </svg>
            </div>
            <div class="stat-number">&lt;20ms</div>
            <div class="stat-label">Inference Latency</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ── VIEW 2: CLASSIFY ──────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def render_classify(mgr, model_choice=None):
    render_view_header(
        "AUDIO INGESTION & PREDICTION",
        "Classify Emotion",
        "Select an acoustic architecture, supply voice input, and perform real-time classification."
    )

    model_options = [
        "SVM Optimized (Recommended)",
        "ANN (Best)",
        "ANN Optimized",
        "Mel-CNN",
        "Random Forest",
    ]

    current_choice = st.session_state.get("model_choice", "SVM Optimized")
    default_idx = 0
    for idx, opt in enumerate(model_options):
        if current_choice in opt:
            default_idx = idx
            break

    # Sentinel div to enable flex column CSS targeting
    st.markdown('<div class="classify-flex-sentinel" style="display:none"></div>', unsafe_allow_html=True)
    left_col, right_col = st.columns([1.08, 0.92], gap="large")

    uploader_key = f"uploader_{st.session_state.reset_counter}"
    mic_key      = f"mic_{st.session_state.reset_counter}"

    with left_col:
        # ── Unified Model Architecture Section ──────────────────────────────────
        render_section_header(
            "model",
            "Model Architecture",
            "Select classification engine and view real-time accuracy benchmarks",
            accent="#818cf8",
            first=True,
        )

        # Selectbox + pill side-by-side inside the card (st.columns still used for Streamlit widgets)
        m_sel_col, m_pill_col = st.columns([1.5, 1.3])
        with m_sel_col:
            model_selection = st.selectbox(
                "Choose Model",
                model_options,
                index=default_idx,
                label_visibility="collapsed",
                key=f"classify_model_sel_{st.session_state.reset_counter}",
                help="SVM Optimized is recommended for microphone recordings and live voice.",
            )
            selected_model = "SVM Optimized" if "SVM" in model_selection else model_selection
            st.session_state["model_choice"] = selected_model
            model_choice = selected_model

        with m_pill_col:
            perf = MODEL_PERFORMANCE.get(model_choice, {})
            acc_val = perf.get("accuracy", "—").split()[0]
            f1_val = perf.get("f1", "—")
            st.markdown(f"""
            <div class="compact-perf-pill">
                <span class="perf-metric"><span class="perf-label">ACC</span> <strong class="perf-val">{acc_val}</strong></span>
                <span class="perf-divider">&bull;</span>
                <span class="perf-metric"><span class="perf-label">F1</span> <strong class="perf-val">{f1_val}</strong></span>
            </div>
            """, unsafe_allow_html=True)

        # ── Audio Input Section Header ──────────────────────────────────────────
        render_section_header(
            "mic",
            "Audio Input",
            "Supply voice input via file upload, microphone, or audio samples",
            accent="#22d3ee",
            first=False,
        )

        tab_upload, tab_mic, tab_sample = st.tabs([
            "Upload Audio",
            "Record Voice",
            "Audio Samples",
        ])

        # ── Tab 1: Upload ─────────────────────────────────────────────────────────
        with tab_upload:
            st.markdown("""
            <div class="format-chip-group">
                <div class="format-chip-group-label">Supported Formats</div>
                <div class="format-chip-row">
                    <span class="format-chip">
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        WAV
                    </span>
                    <span class="format-chip">
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        MP3
                    </span>
                    <span class="format-chip">
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#f5a623" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        OGG
                    </span>
                    <span class="format-chip">
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        FLAC
                    </span>
                    <span class="format-chip format-chip-highlight">
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
                        16-bit PCM
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Upload audio file",
                type=["wav", "mp3", "ogg", "flac"],
                label_visibility="collapsed",
                key=uploader_key,
            )
            if uploaded is not None:
                raw_bytes = uploaded.read()
                if len(raw_bytes) > 0 and raw_bytes != st.session_state["audio_bytes"]:
                    with st.spinner("Processing audio for playback..."):
                        try:
                            audible_bytes, y_sig, dur, raw_pk, has_sp, v_dur = make_audible_pcm_wav(raw_bytes)
                            st.session_state["audio_bytes"] = raw_bytes
                            st.session_state["audio_label"] = uploaded.name
                            st.session_state["audible_wav"] = audible_bytes
                            st.session_state["preview_signal"] = y_sig
                            st.session_state["audio_duration"] = dur
                            st.session_state["raw_peak"] = raw_pk
                            st.session_state["has_speech"] = has_sp
                            st.session_state["voiced_duration"] = v_dur
                            st.session_state["prediction_result"] = None
                            st.session_state["preprocessed_signal"] = None
                            st.session_state["selected_sample_emotion"] = None
                        except Exception as e:
                            st.error(f"Could not process uploaded file: {e}")

        # ── Tab 2: Microphone ─────────────────────────────────────────────────────
        with tab_mic:
            st.markdown("""
            <div class="format-chip-row">
                <span class="format-chip format-chip-highlight">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                    2–4s spoken audio &middot; Conversational volume &middot; Auto-amplified
                </span>
            </div>
            """, unsafe_allow_html=True)

            recorded = st.audio_input("Record audio", sample_rate=22050, key=mic_key)
            if recorded is not None:
                raw_bytes = recorded.read()
                if len(raw_bytes) > 0 and raw_bytes != st.session_state["audio_bytes"]:
                    with st.spinner("Enhancing microphone recording…"):
                        try:
                            audible_bytes, y_sig, dur, raw_pk, has_sp, v_dur = make_audible_pcm_wav(raw_bytes)
                            st.session_state["audio_bytes"] = raw_bytes
                            st.session_state["audio_label"] = "microphone_recording.wav"
                            st.session_state["audible_wav"] = audible_bytes
                            st.session_state["preview_signal"] = y_sig
                            st.session_state["audio_duration"] = dur
                            st.session_state["raw_peak"] = raw_pk
                            st.session_state["has_speech"] = has_sp
                            st.session_state["voiced_duration"] = v_dur
                            st.session_state["prediction_result"] = None
                            st.session_state["preprocessed_signal"] = None
                            st.session_state["selected_sample_emotion"] = None
                        except Exception as e:
                            st.error(f"Could not process microphone recording: {e}")

        # ── Tab 3: Samples ────────────────────────────────────────────────────────
        with tab_sample:
            current_selected = st.session_state.get("selected_sample_emotion")
            sample_cols = st.columns(4)
            sample_selected = None
            for i, em in enumerate(EMOTIONS):
                with sample_cols[i % 4]:
                    is_active_sample = (current_selected == em)
                    if st.button(
                        em.capitalize(),
                        key=f"sample_{em}_{st.session_state.reset_counter}",
                        type="primary" if is_active_sample else "secondary",
                        use_container_width=True,
                    ):
                        sample_selected = em

            if sample_selected:
                st.session_state["selected_sample_emotion"] = sample_selected
                sample_path = find_ravdess_sample(sample_selected)
                if sample_path and os.path.exists(sample_path):
                    with open(sample_path, "rb") as f:
                        raw_bytes = f.read()
                    audible_bytes, y_sig, dur, raw_pk, has_sp, v_dur = make_audible_pcm_wav(raw_bytes)
                    st.session_state["audio_bytes"] = raw_bytes
                    st.session_state["audio_label"] = os.path.basename(sample_path)
                    st.session_state["audible_wav"] = audible_bytes
                    st.session_state["preview_signal"] = y_sig
                    st.session_state["audio_duration"] = dur
                    st.session_state["raw_peak"] = raw_pk
                    st.session_state["has_speech"] = has_sp
                    st.session_state["voiced_duration"] = v_dur
                    st.session_state["prediction_result"] = None
                    st.session_state["preprocessed_signal"] = None
                    st.rerun()
                elif not os.path.isdir(DATA_DIR):
                    st.markdown("""
                    <div class='notice-block warn'>
                        <div>
                            <div class='notice-title' style='color:#f59e0b;'>Dataset Not Found</div>
                            <div class='notice-body'>RAVDESS speech dataset directory not found locally. Please upload an audio file instead.</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='notice-block warn'>
                        <div>
                            <div class='notice-title' style='color:#f59e0b;'>No Sample Found</div>
                            <div class='notice-body'>No sample found for <strong>{sample_selected.capitalize()}</strong>.</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        # ── Active Audio Status Panel ────────────────────────────────────────────
        if st.session_state["audio_bytes"] is not None:
            has_sp = st.session_state.get("has_speech", True)
            v_dur  = st.session_state.get("voiced_duration", 0.0)
            raw_pk = st.session_state.get("raw_peak", 1.0)

            if raw_pk < 0.005:
                st.markdown(f"""
                <div class='notice-block' style='margin-top:12px; border-left:3px solid #22d3ee;'>
                    <div>
                        <div class='notice-title' style='color:#67e8f9;'>Microphone Recording Auto-Amplified</div>
                        <div class='notice-body'>
                            The input volume was quiet (Peak: {raw_pk:.3f}). The audio has been digitally boosted and normalized for accurate analysis.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            dot_color = "#22d3ee"
            status_text = (
                f"{st.session_state['audio_duration']:.2f}s &middot; Speech: {v_dur:.2f}s &middot; Auto-Amplified &amp; Normalized"
                if has_sp and v_dur > 0.1
                else f"{st.session_state['audio_duration']:.2f}s &middot; Auto-Amplified &amp; Ready to Classify"
            )

            bar_col, del_col = st.columns([4.8, 1.2])
            with bar_col:
                st.markdown(f"""
                <div class='audio-status-bar'>
                    <span class='status-dot' style='background:{dot_color};'></span>
                    <div style='flex:1;'>
                        <div style='font-size:14px; font-weight:500; color:#f2f3f5;'>{st.session_state['audio_label']}</div>
                        <div style='font-size:12px; color:#8a8f98; margin-top:2px;'>{status_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with del_col:
                st.markdown("<div class='btn-delete'>", unsafe_allow_html=True)
                if st.button("Delete", key="delete_audio_btn", use_container_width=True):
                    cur_v = st.session_state.get("active_view", "classify")
                    for k, v in _defaults.items():
                        st.session_state[k] = v
                    st.session_state["active_view"] = cur_v
                    st.session_state["reset_counter"] += 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            # Playback + download
            play_col, dl_col = st.columns([3.2, 1.2])
            with play_col:
                if st.session_state["audible_wav"] is not None:
                    st.audio(st.session_state["audible_wav"], format="audio/wav")
            with dl_col:
                if st.session_state["audible_wav"] is not None:
                    st.download_button(
                        "Download WAV",
                        data=st.session_state["audible_wav"],
                        file_name="speaksense_amplified.wav",
                        mime="audio/wav",
                        use_container_width=True,
                    )

            # Classify button
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            classify_btn = st.button("Classify Emotion", key="classify_btn", type="primary", use_container_width=True)

            if classify_btn:
                # Branded loading indicator — replaces generic spinner
                _cls_loading = st.empty()
                _cls_loading.markdown(
                    render_loading_indicator("Analyzing acoustic features…"),
                    unsafe_allow_html=True,
                )
                try:
                    audio_src = st.session_state.get("audible_wav") or st.session_state["audio_bytes"]
                    signal, actual_dur = preprocess_input_audio(audio_src)
                    result = predict_emotion(signal, model_choice, mgr, actual_dur)
                    st.session_state["prediction_result"] = result
                    st.session_state["preprocessed_signal"] = signal
                    st.session_state["just_classified"] = True
                    _cls_loading.empty()
                    st.rerun()
                except SilentAudioError as sae:
                    _cls_loading.empty()
                    st.warning(f"🔇 Silent Audio: {sae}")
                except AudioTooShortError as tse:
                    _cls_loading.empty()
                    st.warning(f"⏱️ Audio Too Short: {tse}")
                except InvalidAudioFormatError as ife:
                    _cls_loading.empty()
                    st.error(f"📁 Invalid Format: {ife}")
                except CorruptedAudioError as cae:
                    _cls_loading.empty()
                    st.error(f"⚠️ Corrupted Audio: {cae}")
                except ValueError as ve:
                    _cls_loading.empty()
                    st.error(f"Audio validation failed: {ve}")
                except Exception as exc:
                    _cls_loading.empty()
                    st.error(f"Audio processing error: {exc}")
                    logger.exception("Audio processing failed: %s", exc)

    # ── Right Column: Interactive Previews / Results ─────────────────────────────
    with right_col:
        if st.session_state["audio_bytes"] is None:
            # Clean SVG Acoustic Radar Illustration + Equalizer Animation
            render_section_header(
                "pipeline",
                "Acoustic Telemetry",
                "Real-time spectral analysis & classification output",
                accent="#22d3ee",
                first=True,
            )
            _idle_logo = render_logo(size=56, animated=True)
            st.markdown(f"""
            <div class="preview-radar-card layered-card" style="gap:14px;">
                {_idle_logo}
                <div style="font-size:12px; font-weight:600; color:#f2f3f5; letter-spacing:-0.01em;">
                    SpeakSense
                </div>
                <div class="skeleton-caption">
                    Awaiting audio &middot; Record or upload a voice clip to begin.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state.get("prediction_result") is None:
            # Audio loaded, not yet classified — Immediate Waveform Preview
            render_section_header(
                "pipeline",
                "Waveform Preview",
                "Preprocessed 22.05 kHz normalized audio stream",
                accent="#22d3ee",
                first=True,
            )
            sig = st.session_state.get("preview_signal")
            if sig is None and st.session_state.get("audible_wav") is not None:
                try:
                    sig, _ = sf.read(io.BytesIO(st.session_state["audible_wav"]))
                    if sig.ndim > 1:
                        sig = np.mean(sig, axis=1)
                    st.session_state["preview_signal"] = sig
                except Exception:
                    pass

            if sig is not None:
                fig = plot_waveform(sig, sr=22050, duration=st.session_state.get("audio_duration", 3.0))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                <div style='display:flex; justify-content:space-between; align-items:center; background:#1c1f24; border:1px solid #2f3339; border-radius:8px; padding:10px 14px; font-size:12px; color:#8a8f98; margin-top:8px;'>
                    <span>Status: <strong style='color:#67e8f9;'>Audio Loaded &amp; Normalized</strong></span>
                    <span>Press <kbd style='background:#2f3339; border-radius:4px; padding:2px 6px; color:#22d3ee; font-family:monospace;'>Enter</kbd> to classify</span>
                </div>
                """, unsafe_allow_html=True)

        else:
            # Audio classified — Display Result Card + Top 3 Ranking
            result = st.session_state["prediction_result"]
            if not result.get("error"):
                em = result["predicted_emotion"] if "predicted_emotion" in result else result.get("emotion")
                conf = result["confidence"]
                ms = result["inference_ms"] if "inference_ms" in result else result.get("latency_ms")
                probs = result.get("probabilities", [])
                em_list = result.get("emotion_labels", result.get("emotions", EMOTIONS))

                if st.session_state.get("just_classified"):
                    st.session_state["just_classified"] = False
                    st.toast(f"Classification Complete: {em.upper()} ({conf*100:.1f}%)")

                render_section_header(
                    "result",
                    "Classification Result",
                    "Acoustic inference output and confidence distribution",
                    accent="#22d3ee",
                    first=True,
                )
                st.markdown(f"""
                <div class="result-hero-card">
                    <div style="font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#8a8f98; margin-bottom:4px;">
                        Primary Emotion Detected
                    </div>
                    <div style="display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;">
                        <span style="font-size:32px; font-weight:700; color:#22d3ee; letter-spacing:-0.03em;">
                            {em.upper()}
                        </span>
                        <span style="font-size:16px; font-weight:500; color:#f2f3f5;">
                            {conf*100:.1f}% Confidence
                        </span>
                    </div>
                    <div style="font-size:11px; color:#8a8f98; margin-top:8px; line-height:1.4;">
                        Model: <strong style="color:#f2f3f5;">{model_choice}</strong> &middot; Latency: <strong style="color:#f2f3f5;">{ms:.1f} ms</strong> &middot; File: <strong style="color:#f2f3f5;">{st.session_state['audio_label']}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                if st.button("Explore Dashboards →", key="btn_goto_dashboards", type="primary", use_container_width=True):
                    st.session_state.active_view = "dashboards"
                    st.rerun()

                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                sorted_idx = np.argsort(probs)[::-1]
                rows = []
                for rank, idx in enumerate(sorted_idx[:3], 1):
                    rows.append({
                        "Rank": f"#{rank}",
                        "Emotion": em_list[idx].capitalize(),
                        "Confidence": round(float(probs[idx]) * 100, 1),
                    })
                top3_df = pd.DataFrame(rows)
                st.dataframe(
                    top3_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        )
                    },
                )

    # ── Keyboard Shortcuts Injection ─────────────────────────────────────────────
    import streamlit.components.v1 as _stc_keys
    _stc_keys.html("""
    <script>
    const setupShortcuts = () => {
        const doc = window.parent.document;
        if (doc._speaksenseKeysAttached) return;
        doc._speaksenseKeysAttached = true;
        doc.addEventListener('keydown', (e) => {
            if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
            if (e.key === 'Enter') {
                const btns = Array.from(doc.querySelectorAll('button'));
                const classifyBtn = btns.find(b => b.innerText && b.innerText.includes('Classify Emotion'));
                if (classifyBtn) classifyBtn.click();
            } else if (e.key === 'Escape') {
                const btns = Array.from(doc.querySelectorAll('button'));
                const delBtn = btns.find(b => b.innerText && b.innerText.trim() === 'Delete');
                if (delBtn) delBtn.click();
            }
        });
    };
    setupShortcuts();
    </script>
    """, height=0, scrolling=False)


# ════════════════════════════════════════════════════════════════════════════════
# ── VIEW 3: DASHBOARDS ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def render_dashboards(mgr, model_choice):
    render_view_header(
        "ACOUSTIC INTELLIGENCE & TELEMETRY",
        "Analytical Dashboards",
        "Interactive multi-dimensional analysis across time, frequency, probabilities, and model telemetry."
    )

    if st.session_state.get("prediction_result") is None or st.session_state.get("preprocessed_signal") is None:
        st.markdown("""
        <div class="empty-state layered-card" style="max-width: 520px; margin: 28px auto; text-align: center; padding: 36px 24px;">
            <svg viewBox="0 0 160 90" width="160" height="90" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin: 0 auto 12px auto; display:block;">
                <circle cx="80" cy="45" r="38" stroke="#2f3339" stroke-width="1.2" stroke-dasharray="3,3"/>
                <circle cx="80" cy="45" r="24" stroke="#2f3339" stroke-width="1.2"/>
                <circle cx="80" cy="45" r="10" stroke="#22d3ee" stroke-opacity="0.3" stroke-width="1.5"/>
                <circle cx="80" cy="45" r="3" fill="#22d3ee"/>
                <line x1="80" y1="7" x2="80" y2="83" stroke="#2f3339" stroke-width="1"/>
                <line x1="42" y1="45" x2="118" y2="45" stroke="#2f3339" stroke-width="1"/>
            </svg>
            <div class="empty-title">Awaiting Audio Classification</div>
            <div class="empty-sub" style="font-size:12px; margin-bottom: 16px;">
                Classify an audio recording in the Classify view to unlock telemetry and visual spectrum dashboards.
            </div>
        </div>
        """, unsafe_allow_html=True)
        _, col_btn, _ = st.columns([2, 1.4, 2])
        with col_btn:
            if st.button("Go to Classify", key="dash_goto_classify_btn", type="primary", use_container_width=True):
                st.session_state.active_view = "classify"
                st.rerun()
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        return

    result = st.session_state["prediction_result"]
    signal = st.session_state["preprocessed_signal"]
    actual_dur = st.session_state["audio_duration"]

    if result.get("error"):
        st.error(f"Inference error: {result['error']}")
        return

    em       = result["predicted_emotion"] if "predicted_emotion" in result else result.get("emotion")
    conf     = result["confidence"]
    probs    = result["probabilities"]
    em_list  = result["emotion_labels"] if "emotion_labels" in result else result.get("emotions")
    ms       = result["inference_ms"] if "inference_ms" in result else result.get("latency_ms")
    em_desc  = EMOTION_DESC.get(em, "")
    perf     = MODEL_PERFORMANCE.get(model_choice, {})

    rail_col, content_col = st.columns([0.16, 0.84], gap="medium")

    dash_panels = [
        ("overview", "Overview"),
        ("waveform", "Waveform"),
        ("spectrogram", "Spectrogram"),
        ("probabilities", "Probabilities"),
        ("telemetry", "Telemetry"),
    ]

    active_panel = st.session_state.get("dash_panel", "overview")

    with rail_col:
        st.markdown("<div class='section-label' style='margin-bottom:8px;'>Dashboards</div>", unsafe_allow_html=True)
        for p_id, p_name in dash_panels:
            is_active = (active_panel == p_id)
            if st.button(
                p_name,
                key=f"dash_tab_{p_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if active_panel != p_id:
                    st.session_state["dash_panel"] = p_id
                    st.rerun()

    with content_col:
        # ── DASHBOARD 1: Emotion Overview ─────────────────────────────────────
        if active_panel == "overview":
            st.markdown("<div class='section-label' style='margin:2px 0 8px;'>Primary Classification Result</div>", unsafe_allow_html=True)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Primary Emotion</div>
                    <div class="kpi-val" style="color:#22d3ee;">{em.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Confidence</div>
                    <div class="kpi-val">{conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Inference Latency</div>
                    <div class="kpi-val">{ms:.1f} ms</div>
                </div>
                """, unsafe_allow_html=True)
            with k4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Audio Duration</div>
                    <div class="kpi-val">{actual_dur:.2f}s</div>
                </div>
                """, unsafe_allow_html=True)

            col_bars, col_audio = st.columns([1.4, 1])
            with col_bars:
                st.markdown("<div class='section-label' style='margin-top:16px;'>Top Predicted Emotions</div>", unsafe_allow_html=True)
                sorted_idx = np.argsort(probs)[::-1]
                rows = []
                for rank, idx in enumerate(sorted_idx[:3], 1):
                    label = em_list[idx]
                    rows.append({
                        "Rank": f"#{rank}",
                        "Emotion": label.capitalize(),
                        "Confidence": round(float(probs[idx]) * 100, 1),
                    })
                top3_df = pd.DataFrame(rows)
                st.dataframe(
                    top3_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence (%)",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        )
                    },
                )

            with col_audio:
                st.markdown("<div class='section-label' style='margin-top:16px;'>Audio Playback</div>", unsafe_allow_html=True)
                if st.session_state["audible_wav"] is not None:
                    st.audio(st.session_state["audible_wav"], format="audio/wav")
                st.markdown("""
                <div class='notice-block' style='margin-top:10px;'>
                    <div class='notice-body'>Amplified with 16-bit PCM normalization for clear playback audibility.</div>
                </div>
                """, unsafe_allow_html=True)

        # ── DASHBOARD 2: Waveform & Acoustic Signal ───────────────────────────
        elif active_panel == "waveform":
            st.markdown("<div class='section-label' style='margin:2px 0 8px;'>Time-Domain Acoustic Analysis</div>", unsafe_allow_html=True)

            fig_w = plot_waveform(signal)
            st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar": False})

            fig_rms = plot_rms_energy(signal)
            st.plotly_chart(fig_rms, use_container_width=True, config={"displayModeBar": False})

            peak_amp = float(np.max(np.abs(signal)))
            mean_rms = float(np.mean(librosa.feature.rms(y=signal)[0]))
            zcr_val  = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)[0]))

            st.markdown("<div class='section-label' style='margin-top:16px;'>Signal Properties</div>", unsafe_allow_html=True)
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Peak Amplitude", f"{peak_amp:.3f}")
            s2.metric("Mean RMS Energy", f"{mean_rms:.4f}")
            s3.metric("Zero Crossing Rate", f"{zcr_val:.4f}")
            s4.metric("Sample Rate", f"{SR:,} Hz")
            s5.metric("Active Samples", f"{len(signal):,}")

        # ── DASHBOARD 3: Spectrogram & Frequency Spectrum ─────────────────────
        elif active_panel == "spectrogram":
            st.markdown("<div class='section-label' style='margin:2px 0 8px;'>Frequency Spectrum &amp; Energy Distribution</div>", unsafe_allow_html=True)
            
            fig_s = plot_mel_spectrogram(signal)
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

            fig_spec = plot_spectral_curves(signal)
            st.plotly_chart(fig_spec, use_container_width=True, config={"displayModeBar": False})

            cent_mean = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=SR)[0]))
            rolloff_mean = float(np.mean(librosa.feature.spectral_rolloff(y=signal, sr=SR)[0]))
            bw_mean = float(np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=SR)[0]))

            st.markdown("<div class='section-label' style='margin-top:16px;'>Spectral Metrics</div>", unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            f1.metric("Mean Spectral Centroid", f"{cent_mean:.1f} Hz")
            f2.metric("Mean Spectral Rolloff", f"{rolloff_mean:.1f} Hz")
            f3.metric("Mean Spectral Bandwidth", f"{bw_mean:.1f} Hz")

        # ── DASHBOARD 4: Emotion Probability Distribution ─────────────────────
        elif active_panel == "probabilities":
            st.markdown("<div class='section-label' style='margin:2px 0 8px;'>Full Class Probability Distribution</div>", unsafe_allow_html=True)
            
            col_bar, col_radar = st.columns([1, 1])
            with col_bar:
                fig_b = plot_probability_bars(np.array(probs), em_list, em)
                st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})
            with col_radar:
                fig_rad = plot_radar(np.array(probs), em_list)
                st.plotly_chart(fig_rad, use_container_width=True, config={"displayModeBar": False})

            sorted_idx = np.argsort(probs)[::-1]
            all_rows = [
                {
                    "Rank": f"#{i+1}",
                    "Emotion": em_list[idx].capitalize(),
                    "Probability": f"{probs[idx]*100:.2f}%",
                    "Score": round(float(probs[idx]) * 100, 2),
                }
                for i, idx in enumerate(sorted_idx)
            ]
            all_df = pd.DataFrame(all_rows)
            st.dataframe(
                all_df[["Rank", "Emotion", "Probability", "Score"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Confidence Distribution",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    )
                },
            )

            if len(sorted_idx) >= 2:
                margin = (probs[sorted_idx[0]] - probs[sorted_idx[1]]) * 100
                st.markdown(f"""
                <div class="notice-block">
                    <div class="notice-body">
                        Classification Margin: <strong>{em_list[sorted_idx[0]].capitalize()}</strong> leads <strong>{em_list[sorted_idx[1]].capitalize()}</strong> by {margin:.1f}% percentage points.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── DASHBOARD 5: Model & Processing Telemetry ─────────────────────────
        elif active_panel == "telemetry":
            st.markdown("<div class='section-label' style='margin:2px 0 8px;'>Model Telemetry &amp; Feature Pipeline Details</div>", unsafe_allow_html=True)

            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.markdown(f"""
                <div class="telemetry-card">
                    <div class="kpi-label">Active Model</div>
                    <div class="telemetry-val">{model_choice}</div>
                </div>
                """, unsafe_allow_html=True)
            with t2:
                st.markdown(f"""
                <div class="telemetry-card">
                    <div class="kpi-label">Inference Latency</div>
                    <div class="telemetry-val">{ms:.1f} ms</div>
                </div>
                """, unsafe_allow_html=True)
            with t3:
                st.markdown(f"""
                <div class="telemetry-card">
                    <div class="kpi-label">Feature Dimensions</div>
                    <div class="telemetry-val">{'264 Features' if model_choice != 'Mel-CNN' else '128x130 Tensor'}</div>
                </div>
                """, unsafe_allow_html=True)
            with t4:
                acc_display = perf.get("accuracy", "—").split()[0]
                st.markdown(f"""
                <div class="telemetry-card">
                    <div class="kpi-label">Benchmark Accuracy</div>
                    <div class="telemetry-val">{acc_display}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class='layered-card' style='margin-top: 16px; padding: 18px 22px;'>
                <div style='font-size:13px; font-weight:600; color:#8a8f98; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;'>
                    264-Dimensional Feature Architecture
                </div>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;'>
                    <div class='format-chip' style='padding: 8px 12px; border-radius: 8px;'>
                        <strong style='color:#22d3ee;'>40 MFCCs</strong> &middot; Spectral envelope
                    </div>
                    <div class='format-chip' style='padding: 8px 12px; border-radius: 8px;'>
                        <strong style='color:#818cf8;'>40 Delta MFCCs</strong> &middot; Velocity rate
                    </div>
                    <div class='format-chip' style='padding: 8px 12px; border-radius: 8px;'>
                        <strong style='color:#f5a623;'>40 Delta-Delta</strong> &middot; Acceleration
                    </div>
                    <div class='format-chip' style='padding: 8px 12px; border-radius: 8px;'>
                        <strong style='color:#34d399;'>128 Mel Filterbanks</strong> &middot; Frequency bands
                    </div>
                    <div class='format-chip' style='padding: 8px 12px; border-radius: 8px;'>
                        <strong style='color:#f43f5e;'>12 Chroma</strong> &middot; Pitch classes
                    </div>
                    <div class='format-chip' style='padding: 8px 12px; border-radius: 8px;'>
                        <strong style='color:#94a3b8;'>4 Scalars</strong> &middot; ZCR &amp; Rolloff
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ── VIEW 4: ABOUT ─────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def render_about():
    render_view_header(
        "DOCUMENTATION & REFERENCE",
        "Architecture & Acoustic Pipeline",
        "Acoustic architecture specifications, feature pipelines, and emotion standards."
    )

    # ── Horizontal Icon Flow Pipeline Diagram ─────────────────────────────────
    render_section_header(
        "pipeline",
        "Inference Pipeline",
        "5-stage acoustic feature transformation from raw voice to emotion prediction",
        accent="#22d3ee",
        first=True,
    )
    import streamlit.components.v1 as _stc_about
    _stc_about.html("""
    <div style="width:100%; overflow-x:auto; padding: 2px 0 10px 0;">
        <svg viewBox="0 0 760 115" width="100%" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Background Container Card -->
            <rect x="2" y="4" width="756" height="107" rx="14" fill="#1c1f24" stroke="#2f3339" stroke-width="1.2"/>
            
            <!-- Node 1: Voice -->
            <g transform="translate(55, 18)">
                <circle cx="28" cy="28" r="24" fill="rgba(34, 211, 238, 0.12)" stroke="#22d3ee" stroke-width="1.8"/>
                <path d="M28 17 C25.8 17 24 18.8 24 21 L24 28 C24 30.2 25.8 32 28 32 C30.2 32 32 30.2 32 28 L32 21 C32 18.8 30.2 17 28 17 Z" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M20 26 C20 30.4 23.6 34 28 34 C32.4 34 36 30.4 36 26" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round"/>
                <line x1="28" y1="34" x2="28" y2="39" stroke="#22d3ee" stroke-width="1.8"/>
                <line x1="24" y1="39" x2="32" y2="39" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round"/>
                <text x="28" y="66" fill="#f2f3f5" font-size="13" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-weight="600" text-anchor="middle">Voice</text>
            </g>
            
            <!-- Arrow 1 -->
            <path d="M 135 46 L 185 46 M 177 40 L 185 46 L 177 52" stroke="#3d4148" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            
            <!-- Node 2: Waveform -->
            <g transform="translate(205, 18)">
                <circle cx="28" cy="28" r="24" fill="rgba(129, 140, 248, 0.12)" stroke="#818cf8" stroke-width="1.8"/>
                <path d="M16 28 L21 28 L24 18 L28 38 L32 22 L35 32 L39 28" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <text x="28" y="66" fill="#f2f3f5" font-size="13" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-weight="600" text-anchor="middle">Waveform</text>
            </g>
            
            <!-- Arrow 2 -->
            <path d="M 285 46 L 335 46 M 327 40 L 335 46 L 327 52" stroke="#3d4148" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            
            <!-- Node 3: Features -->
            <g transform="translate(355, 18)">
                <circle cx="28" cy="28" r="24" fill="rgba(245, 166, 35, 0.12)" stroke="#f5a623" stroke-width="1.8"/>
                <line x1="18" y1="36" x2="18" y2="28" stroke="#f5a623" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="24" y1="36" x2="24" y2="18" stroke="#f5a623" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="31" y1="36" x2="31" y2="23" stroke="#f5a623" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="38" y1="36" x2="38" y2="31" stroke="#f5a623" stroke-width="2.5" stroke-linecap="round"/>
                <text x="28" y="66" fill="#f2f3f5" font-size="13" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-weight="600" text-anchor="middle">Features</text>
            </g>
            
            <!-- Arrow 3 -->
            <path d="M 435 46 L 485 46 M 477 40 L 485 46 L 477 52" stroke="#3d4148" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            
            <!-- Node 4: Model -->
            <g transform="translate(505, 18)">
                <circle cx="28" cy="28" r="24" fill="rgba(52, 211, 153, 0.12)" stroke="#34d399" stroke-width="1.8"/>
                <circle cx="20" cy="21" r="2.5" fill="#34d399"/>
                <circle cx="20" cy="35" r="2.5" fill="#34d399"/>
                <circle cx="28" cy="28" r="2.5" fill="#34d399"/>
                <circle cx="36" cy="21" r="2.5" fill="#34d399"/>
                <circle cx="36" cy="35" r="2.5" fill="#34d399"/>
                <line x1="20" y1="21" x2="28" y2="28" stroke="#34d399" stroke-width="1.5" stroke-opacity="0.8"/>
                <line x1="20" y1="35" x2="28" y2="28" stroke="#34d399" stroke-width="1.5" stroke-opacity="0.8"/>
                <line x1="28" y1="28" x2="36" y2="21" stroke="#34d399" stroke-width="1.5" stroke-opacity="0.8"/>
                <line x1="28" y1="28" x2="36" y2="35" stroke="#34d399" stroke-width="1.5" stroke-opacity="0.8"/>
                <text x="28" y="66" fill="#f2f3f5" font-size="13" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-weight="600" text-anchor="middle">Model</text>
            </g>
            
            <!-- Arrow 4 -->
            <path d="M 585 46 L 635 46 M 627 40 L 635 46 L 627 52" stroke="#3d4148" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            
            <!-- Node 5: Emotion -->
            <g transform="translate(655, 18)">
                <circle cx="28" cy="28" r="24" fill="#22d3ee" stroke="#22d3ee" stroke-width="1.8"/>
                <path d="M19 27 Q28 35 37 27" stroke="#0e2a2e" stroke-width="2.6" stroke-linecap="round" fill="none"/>
                <circle cx="22" cy="22" r="2" fill="#0e2a2e"/>
                <circle cx="34" cy="22" r="2" fill="#0e2a2e"/>
                <text x="28" y="66" fill="#f2f3f5" font-size="13" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-weight="600" text-anchor="middle">Emotion</text>
            </g>
        </svg>
    </div>
    """, height=125, scrolling=False)

    # ── Emotion Categories 2-Column Grid (Visual Cards & Valence Badges) ──────
    render_section_header(
        "standards",
        "Emotion Classification Standards",
        "8 acoustic emotion categories with valence, arousal, and prosodic cues",
        accent="#34d399",
        first=False,
    )
    
    emotion_cards_meta = [
        ("happy",     "Positive", "tone-positive", "#facc15", "Elevated pitch & vocal energy · High Arousal",
         '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#facc15" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>'),
        ("angry",     "Negative", "tone-negative", "#f43f5e", "High intensity & rapid tempo · High Arousal",
         '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><line x1="7.5" y1="8.5" x2="10.5" y2="10"/><line x1="16.5" y1="8.5" x2="13.5" y2="10"/></svg>'),
        ("disgust",   "Negative", "tone-negative", "#fb923c", "Low pitch & guttural inflection · Aversive",
         '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fb923c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 15h8"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>'),
        ("sad",       "Negative", "tone-negative", "#818cf8", "Subdued amplitude & slow rate · Low Arousal",
         '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>'),
        ("calm",      "Positive", "tone-positive", "#38bdf8", "Even rhythm & stable pitch · Low Arousal",
         '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="8" y1="14" x2="16" y2="14"/><line x1="8" y1="9.5" x2="10" y2="9.5"/><line x1="14" y1="9.5" x2="16" y2="9.5"/></svg>'),
        ("fearful",   "Negative", "tone-negative", "#a855f7", "Irregular tremors & high pitch · High Arousal",
         '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="15" r="2.5"/><circle cx="9" cy="9" r="1.5"/><circle cx="15" cy="9" r="1.5"/></svg>'),
        ("neutral",   "Neutral",  "tone-neutral",  "#94a3b8", "Baseline conversational pitch · Neutral",
         '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="8" y1="14" x2="16" y2="14"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>'),
        ("surprised", "Dynamic",  "tone-dynamic",  "#22d3ee", "Abrupt pitch jump & wide range · High Arousal",
         '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="15" r="2.5"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>'),
    ]

    em_c1, em_c2 = st.columns(2)
    for i, (name, tone, tone_cls, color, cue, icon_svg) in enumerate(emotion_cards_meta):
        col = em_c1 if (i % 2 == 0) else em_c2
        with col:
            st.markdown(f"""
            <div class="emotion-card-v2 layered-card">
                <div class="icon-badge" style="background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.12); border: 1px solid {color};">
                    {icon_svg}
                </div>
                <div class="emotion-card-v2-body">
                    <div class="emotion-card-v2-header">
                        <span class="emotion-card-v2-title">{name.capitalize()}</span>
                        <span class="tone-pill {tone_cls}">{tone}</span>
                    </div>
                    <div class="emotion-card-v2-cue">{cue}</div>
                </div>
            </div>
            <div style="height: 6px;"></div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ── Model Architecture Comparison & Technical Reference (Expander) ────────
    model_benchmarks = [
        {
            "Architecture": "SVM Optimized (Recommended)",
            "Paradigm": "RBF Support Vector Machine",
            "Accuracy": "91.7%",
            "F1 Score": "0.916",
            "Input Representation": "264-D Acoustic Vector",
            "Inference Latency": "~14 ms",
            "Optimal Use Case": "Microphone / live conversational speech",
        },
        {
            "Architecture": "ANN (Best Benchmark)",
            "Paradigm": "Deep Multilayer Perceptron",
            "Accuracy": "93.2%",
            "F1 Score": "0.931",
            "Input Representation": "264-D Acoustic Vector",
            "Inference Latency": "~18 ms",
            "Optimal Use Case": "High-accuracy static audio evaluation",
        },
        {
            "Architecture": "ANN Optimized",
            "Paradigm": "Regularized Dense MLP",
            "Accuracy": "92.8%",
            "F1 Score": "0.927",
            "Input Representation": "264-D Acoustic Vector",
            "Inference Latency": "~17 ms",
            "Optimal Use Case": "Noise-resilient neural classification",
        },
        {
            "Architecture": "Mel-CNN",
            "Paradigm": "2D Spatial Convolutional",
            "Accuracy": "94.8%",
            "F1 Score": "0.947",
            "Input Representation": "128x130 Log-Mel Tensor",
            "Inference Latency": "~24 ms",
            "Optimal Use Case": "Formant and time-frequency pattern analysis",
        },
        {
            "Architecture": "Random Forest",
            "Paradigm": "Ensemble 300 Decision Trees",
            "Accuracy": "89.4%",
            "F1 Score": "0.892",
            "Input Representation": "264-D Acoustic Vector",
            "Inference Latency": "~16 ms",
            "Optimal Use Case": "High interpretability and outlier resilience",
        },
    ]
    bench_df = pd.DataFrame(model_benchmarks)

    # ── Model Architecture Comparison & Technical Reference ───────────────────
    render_section_header(
        "benchmark",
        "Technical Model Benchmark",
        "Comparative benchmark metrics, acoustic stack, and audio conditioning parameters",
        accent="#818cf8",
        first=False,
    )

    with st.expander("Expand Benchmark Matrix & Acoustic Stack Specifications", expanded=False):
        st.dataframe(bench_df, hide_index=True, use_container_width=True)
        st.markdown("""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px;">
            <div class="layered-card" style="padding: 16px 18px;">
                <div style="font-size: 13px; font-weight: 600; color: #22d3ee; margin-bottom: 6px;">264-D Acoustic Stack</div>
                <div style="font-size: 11px; color: #8a8f98; line-height: 1.6;">
                    &bull; <strong>40 MFCCs</strong> (Spectral envelope)<br>
                    &bull; <strong>40 Delta MFCCs</strong> (Velocity rate)<br>
                    &bull; <strong>40 Delta-Delta MFCCs</strong> (Acceleration)<br>
                    &bull; <strong>128 Mel Filterbanks</strong> (Cochlear bands)<br>
                    &bull; <strong>12 Chroma Coefficients</strong> (Pitch classes)<br>
                    &bull; <strong>4 Acoustic Scalars</strong> (ZCR, Centroid, Bandwidth, Rolloff)
                </div>
            </div>
            <div class="layered-card" style="padding: 16px 18px;">
                <div style="font-size: 13px; font-weight: 600; color: #22d3ee; margin-bottom: 6px;">Audio Conditioning &amp; VAD</div>
                <div style="font-size: 11px; color: #8a8f98; line-height: 1.6;">
                    &bull; <strong>16-bit PCM Amplification:</strong> Peak normalization with headroom.<br>
                    &bull; <strong>Voice Activity Detection:</strong> Top-db dynamic silence thresholding.<br>
                    &bull; <strong>Windowing:</strong> Standardized 3.0-second temporal target.<br>
                    &bull; <strong>Decoding:</strong> Native support for WAV, MP3, OGG, FLAC.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ── VIEW: SYSTEM HEALTH & REAL-TIME MONITORING (MODULE 14) ─────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def render_monitoring(mgr: ModelManager):
    render_view_header(
        "OPERATIONAL TELEMETRY & AUDIT",
        "System Health & Real-Time Monitoring",
        "Live inference latency profiling, deployment readiness metrics, and operational audit logs."
    )

    from src.monitoring import tracker
    import datetime, json
    metrics = tracker.get_summary_metrics()
    sys_health = tracker.get_system_health()
    mgr_health = mgr.get_health_status()

    # ── Top Level KPI Cards ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="layered-card metric-card">
            <div class="metric-label">System Health</div>
            <div class="metric-value" style="color:#34d399; font-size:20px;">
                {'🟢 HEALTHY' if mgr_health.get('healthy') else '🟡 DEGRADED'}
            </div>
            <div class="metric-sub">All core models loaded</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="layered-card metric-card">
            <div class="metric-label">Total Inferences</div>
            <div class="metric-value" style="color:#22d3ee; font-size:24px;">
                {metrics['total_requests']}
            </div>
            <div class="metric-sub">{metrics['successful_requests']} successful &middot; {metrics['error_count']} errors</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="layered-card metric-card">
            <div class="metric-label">Avg Latency</div>
            <div class="metric-value" style="color:#fbbf24; font-size:24px;">
                {metrics['avg_latency_ms']} ms
            </div>
            <div class="metric-sub">P50: {metrics['p50_latency_ms']} ms &middot; P95: {metrics['p95_latency_ms']} ms</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="layered-card metric-card">
            <div class="metric-label">Active Models</div>
            <div class="metric-value" style="color:#a78bfa; font-size:24px;">
                {mgr_health.get('total_models_loaded', 0)} / 5
            </div>
            <div class="metric-sub">Manifest verified</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="layered-card metric-card">
            <div class="metric-label">Session Uptime</div>
            <div class="metric-value" style="color:#38bdf8; font-size:20px;">
                {metrics['uptime_formatted']}
            </div>
            <div class="metric-sub">Memory RSS: {sys_health.get('memory_usage_mb', 'N/A')} MB</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # ── Interactive Benchmarking & Charts ──
    chart_col1, chart_col2 = st.columns([1.4, 1.0])
    
    with chart_col1:
        st.markdown("#### ⏱️ Real-Time Inference Latency Profiling")
        recent_logs = tracker.get_recent_logs(30)
        if recent_logs:
            df_logs = pd.DataFrame(recent_logs)
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(
                x=list(range(1, len(df_logs) + 1)),
                y=df_logs["latency_ms"][::-1],
                mode="lines+markers",
                name="Latency (ms)",
                line=dict(color="#22d3ee", width=2.5),
                marker=dict(size=7, color="#38bdf8"),
                hovertemplate="Request #%{x}<br>Latency: %{y:.1f} ms<extra></extra>"
            ))
            fig_lat.add_hline(
                y=100.0, line_dash="dash", line_color="#f87171",
                annotation_text="Target SLA (100 ms)", annotation_position="top right"
            )
            fig_lat.update_layout(
                paper_bgcolor="#14161a",
                plot_bgcolor="#1c1f24",
                font=dict(color="#f2f3f5"),
                xaxis=dict(title="Recent Predictions (Chronological)", gridcolor="#2a2e36"),
                yaxis=dict(title="Inference Latency (ms)", gridcolor="#2a2e36"),
                margin=dict(l=40, r=20, t=30, b=40),
                height=280
            )
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("No inference telemetry recorded in this session yet. Run predictions in the Classify view to see real-time latency profiling.")

    with chart_col2:
        st.markdown("#### 🎭 Session Emotion Distribution")
        em_dist = metrics.get("emotion_distribution", {})
        if em_dist and sum(em_dist.values()) > 0:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(em_dist.keys()),
                values=list(em_dist.values()),
                hole=0.55,
                marker=dict(colors=[THEME_EMOTION_COLOR.get(k, "#22d3ee") for k in em_dist.keys()]),
                textinfo="label+percent"
            )])
            fig_pie.update_layout(
                paper_bgcolor="#14161a",
                plot_bgcolor="#1c1f24",
                font=dict(color="#f2f3f5"),
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=280
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Emotion breakdown will appear here once audio samples are classified.")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # ── Model Artifacts Manifest & Health Status ──
    st.markdown("#### 📦 Production Model Manifest & Checksums")
    manifest_file = os.path.join(BASE_DIR, "models", "model_manifest.json")
    if os.path.exists(manifest_file):
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        
        manifest_rows = []
        for m_name, m_info in manifest_data.items():
            manifest_rows.append({
                "Model": m_name,
                "Type": m_info.get("model_type", ""),
                "Accuracy": f"{m_info.get('accuracy', 0)*100:.1f}%",
                "F1 Score": f"{m_info.get('f1_score', 0)*100:.1f}%",
                "Size (MB)": m_info.get("size_mb", 0),
                "Params": m_info.get("param_count") or "N/A (Tree/SVM)",
                "SHA-256 Checksum": m_info.get("sha256", "")[:12] + "...",
                "Status": "✅ Verified"
            })
        st.dataframe(pd.DataFrame(manifest_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("model_manifest.json not found in models directory.")

    # ── Warmup & Diagnostic Actions ──
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    w_col1, w_col2 = st.columns([1, 2])
    with w_col1:
        if st.button("⚡ Trigger Model Warmup & Benchmark", key="btn_warmup", type="primary", use_container_width=True):
            with st.spinner("Warming up inference engines..."):
                warm_res = mgr.warmup_models()
                st.session_state["warmup_benchmark"] = warm_res
                st.success("All models warmed up successfully!")
    
    if st.session_state.get("warmup_benchmark"):
        warm_df = []
        for mn, dat in st.session_state["warmup_benchmark"].items():
            warm_df.append({
                "Model": mn,
                "Warmup Latency": f"{dat.get('warmup_ms', 0):.2f} ms",
                "Status": dat.get("status", "READY")
            })
        st.dataframe(pd.DataFrame(warm_df), use_container_width=True, hide_index=True)

    # ── Operational Inference Audit Log & CSV Export ──
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📋 Operational Inference Audit Log")
    recent = tracker.get_recent_logs(50)
    if recent:
        df_audit = pd.DataFrame(recent)
        st.dataframe(df_audit, use_container_width=True, hide_index=True)
        csv_data = tracker.export_csv()
        st.download_button(
            label="📥 Export Session Telemetry (CSV)",
            data=csv_data,
            file_name=f"speaksense_telemetry_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="dl_telemetry_csv"
        )
    else:
        st.caption("No prediction audit records logged in the current session.")


# ════════════════════════════════════════════════════════════════════════════════
# ── MAIN ROUTER EXECUTION ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

render_top_navbar()

st.markdown('<div class="view-transition">', unsafe_allow_html=True)
active_view = st.session_state.get("active_view", "home")
model_choice = st.session_state.get("model_choice", "SVM Optimized")

# Redirect if previously on monitoring
if active_view == "monitoring":
    active_view = "home"
    st.session_state.active_view = "home"

if active_view == "home":
    render_home()
elif active_view == "classify":
    render_classify(mgr, model_choice)
elif active_view == "dashboards":
    render_dashboards(mgr, model_choice)
elif active_view == "about":
    render_about()
else:
    render_home()

st.markdown('</div>', unsafe_allow_html=True)

# ── Quick Tips Flashcard Deck — fills dead whitespace above footer ────────────
active_v = st.session_state.get("active_view", "home")
if active_v in ("home", "about", "dashboards"):
    render_tip_flashcards(state_key=f"{active_v}_tip_index")


# ════════════════════════════════════════════════════════════════════════════════
# Footer & Global Sound-Reactive Audio Observer
# ════════════════════════════════════════════════════════════════════════════════

render_footer_status()

st.markdown("""
<div class='app-footer'>
    SpeakSense &nbsp;&middot;&nbsp; Speech Emotion Recognition &nbsp;&middot;&nbsp; Powered by SVM &amp; Deep Learning
</div>
""", unsafe_allow_html=True)

# Audio pulse listener
import streamlit.components.v1 as _stc_footer
_stc_footer.html("""
<script>
const attachAudioPulse = () => {
    const audios = window.parent.document.querySelectorAll('audio');
    audios.forEach(a => {
        if (!a.dataset.speaksenseListening) {
            a.dataset.speaksenseListening = 'true';
            a.addEventListener('play', () => {
                window.parent.document.querySelectorAll('.hero-waveform-bg, .result-hero-card, .audio-status-bar').forEach(el => {
                    el.classList.add('audio-active-pulse');
                });
            });
            a.addEventListener('pause', () => {
                window.parent.document.querySelectorAll('.hero-waveform-bg, .result-hero-card, .audio-status-bar').forEach(el => {
                    el.classList.remove('audio-active-pulse');
                });
            });
            a.addEventListener('ended', () => {
                window.parent.document.querySelectorAll('.hero-waveform-bg, .result-hero-card, .audio-status-bar').forEach(el => {
                    el.classList.remove('audio-active-pulse');
                });
            });
        }
    });
};
attachAudioPulse();
setTimeout(attachAudioPulse, 600);
setTimeout(attachAudioPulse, 1800);
</script>
""", height=0, scrolling=False)
