"""
tests/test_invalid_audio.py
───────────────────────────
Module 14 Automated Test Suite: Invalid Audio File Handling & Resilience.
Validates that invalid, silent, corrupted, truncated, and malformed audio inputs
are caught gracefully with descriptive custom exceptions without crashing the system.
"""

import io
import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predictor import (
    preprocess_input_audio,
    AudioProcessingError,
    SilentAudioError,
    AudioTooShortError,
    InvalidAudioFormatError,
    CorruptedAudioError,
    SR,
    DURATION
)
from src.audio_preprocessing import validate_audio_signal


class TestInvalidAudioHandling(unittest.TestCase):
    """Test suite for edge case & invalid audio handling in Module 14."""

    @classmethod
    def setUpClass(cls):
        cls.valid_sample_path = os.path.join(
            BASE_DIR, "Data", "Raw", "Audio_Speech_Actors_01-24", "Actor_01", "03-01-01-01-01-01-01.wav"
        )

    def test_empty_bytes_raises_invalid_format(self):
        """Empty audio bytes must raise InvalidAudioFormatError."""
        with self.assertRaises(InvalidAudioFormatError) as ctx:
            preprocess_input_audio(b"")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_pure_silence_array_raises_silent_audio(self):
        """All-zero signal array must raise SilentAudioError."""
        silent_signal = np.zeros(SR * 2, dtype=np.float32)
        with self.assertRaises(SilentAudioError) as ctx:
            preprocess_input_audio(silent_signal)
        self.assertIn("silent", str(ctx.exception).lower())

    def test_low_amplitude_noise_floor_raises_silent_audio(self):
        """Extremely low amplitude signal (< 1e-6) must raise SilentAudioError."""
        subaudible_signal = np.full(SR * 2, 1e-8, dtype=np.float32)
        with self.assertRaises(SilentAudioError) as ctx:
            preprocess_input_audio(subaudible_signal)
        self.assertIn("silent", str(ctx.exception).lower())

    def test_too_short_signal_raises_audio_too_short(self):
        """Audio clip under 0.25 seconds must raise AudioTooShortError."""
        short_signal = np.random.uniform(-0.5, 0.5, size=int(SR * 0.1)).astype(np.float32)
        with self.assertRaises(AudioTooShortError) as ctx:
            preprocess_input_audio(short_signal)
        self.assertIn("too short", str(ctx.exception).lower())

    def test_corrupted_raw_bytes_raises_corrupted_audio(self):
        """Malformed or non-audio binary data must raise CorruptedAudioError or InvalidAudioFormatError."""
        corrupt_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00CORRUPT_PAYLOAD_GARBAGE"
        with self.assertRaises((CorruptedAudioError, InvalidAudioFormatError)):
            preprocess_input_audio(corrupt_bytes)

    def test_nan_values_in_signal_raises_corrupted_audio(self):
        """Audio containing NaN values must raise CorruptedAudioError."""
        nan_signal = np.random.uniform(-0.5, 0.5, size=SR * 2).astype(np.float32)
        nan_signal[100:150] = np.nan
        with self.assertRaises(CorruptedAudioError) as ctx:
            preprocess_input_audio(nan_signal)
        self.assertIn("nan", str(ctx.exception).lower())

    def test_infinite_values_in_signal_raises_corrupted_audio(self):
        """Audio containing infinite values must raise CorruptedAudioError."""
        inf_signal = np.random.uniform(-0.5, 0.5, size=SR * 2).astype(np.float32)
        inf_signal[200] = np.inf
        with self.assertRaises(CorruptedAudioError) as ctx:
            preprocess_input_audio(inf_signal)
        self.assertIn("infinity", str(ctx.exception).lower())

    def test_validate_audio_signal_helper(self):
        """Verify helper validation function returns accurate flags and messages."""
        # 1. Empty signal
        valid, msg = validate_audio_signal(np.array([]))
        self.assertFalse(valid)
        self.assertIn("empty", msg.lower())

        # 2. Silence
        valid, msg = validate_audio_signal(np.zeros(20000, dtype=np.float32))
        self.assertFalse(valid)
        self.assertIn("silent", msg.lower())

        # 3. Valid signal
        valid, msg = validate_audio_signal(np.random.uniform(-0.3, 0.3, size=22050).astype(np.float32))
        self.assertTrue(valid)
        self.assertEqual(msg, "Valid")

    def test_valid_audio_file_succeeds(self):
        """Standard valid RAVDESS audio file processes to exactly 66150 samples."""
        if os.path.exists(self.valid_sample_path):
            signal, duration = preprocess_input_audio(self.valid_sample_path)
            self.assertEqual(len(signal), int(SR * DURATION))
            self.assertGreater(duration, 0.5)
            self.assertEqual(signal.dtype, np.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
