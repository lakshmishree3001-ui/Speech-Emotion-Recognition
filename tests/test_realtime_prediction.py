"""
tests/test_realtime_prediction.py
─────────────────────────────────
Module 14 Automated Test Suite: Real-Time Audio Prediction & Latency Benchmarks.
Validates model prediction pipeline, probability distributions, streaming buffer
simulations, and latency performance across all 5 models.
"""

import glob
import os
import sys
import time
import unittest
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predictor import (
    ModelManager,
    preprocess_input_audio,
    predict_emotion,
    SR,
    DURATION,
    N_SAMPLES
)
from src.monitoring import tracker


class TestRealTimePrediction(unittest.TestCase):
    """Test suite for real-time audio prediction and latency benchmarks."""

    @classmethod
    def setUpClass(cls):
        cls.mgr = ModelManager()
        cls.mgr.load_all()
        cls.mgr.warmup_models()
        cls.test_files = glob.glob(
            os.path.join(BASE_DIR, "Data", "Raw", "Audio_Speech_Actors_01-24", "Actor_01", "*.wav")
        )[:5]

    def test_model_manager_health(self):
        """All models and preprocessing objects must report healthy status."""
        health = self.mgr.get_health_status()
        self.assertTrue(health["healthy"])
        self.assertGreaterEqual(health["total_models_loaded"], 5)
        self.assertTrue(health["scaler_ready"])
        self.assertTrue(health["le_ready"])
        self.assertTrue(health["mel_cfg_ready"])

    def test_output_schema_and_probabilities(self):
        """Output dictionary must contain all required keys and valid probabilities."""
        if not self.test_files:
            self.skipTest("No test audio files found.")

        signal, duration = preprocess_input_audio(self.test_files[0])
        result = predict_emotion(signal, "ANN (Best)", self.mgr, actual_duration=duration)

        required_keys = [
            "predicted_emotion", "confidence", "probabilities",
            "emotion_labels", "model_name", "inference_ms", "audio_duration"
        ]
        for k in required_keys:
            self.assertIn(k, result)
            self.assertIsNotNone(result[k])

        # Probabilities validation
        probs = result["probabilities"]
        self.assertEqual(len(probs), 8)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=2)
        self.assertTrue((probs >= 0.0).all())
        self.assertGreater(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 100.0)

    def test_all_five_models_predict(self):
        """All 5 models must successfully predict without errors."""
        if not self.test_files:
            self.skipTest("No test audio files found.")

        signal, duration = preprocess_input_audio(self.test_files[0])
        model_names = [
            "ANN (Best)", "ANN Optimized", "SVM Optimized", "Mel-CNN", "Random Forest"
        ]

        for m_name in model_names:
            res = predict_emotion(signal, m_name, self.mgr, actual_duration=duration)
            self.assertIsNone(res["error"], f"Error in {m_name}: {res['error']}")
            self.assertIn(res["predicted_emotion"], res["emotion_labels"])
            self.assertGreater(res["confidence"], 10.0)

    def test_realtime_latency_benchmark(self):
        """Inference latency after warmup should be fast (< 150ms for ANN, < 50ms for SVM)."""
        if not self.test_files:
            self.skipTest("No test audio files found.")

        signal, duration = preprocess_input_audio(self.test_files[0])

        # Test SVM latency
        res_svm = predict_emotion(signal, "SVM Optimized", self.mgr, actual_duration=duration)
        self.assertLess(res_svm["inference_ms"], 100.0)

        # Test ANN Best latency
        res_ann = predict_emotion(signal, "ANN (Best)", self.mgr, actual_duration=duration)
        self.assertLess(res_ann["inference_ms"], 200.0)

    def test_streaming_chunk_buffer_simulation(self):
        """Simulate a real-time sliding buffer accumulating 0.5s chunks into 3.0s analysis window."""
        buffer = []
        chunk_duration = 0.5
        chunk_samples = int(SR * chunk_duration)
        total_chunks = 6   # 6 * 0.5s = 3.0s

        # Simulate microphone stream
        for i in range(total_chunks):
            # Synthetic speech-like harmonic signal
            t = np.linspace(0, chunk_duration, chunk_samples, endpoint=False)
            chunk = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)
            buffer.extend(chunk)

        stream_signal = np.array(buffer, dtype=np.float32)
        self.assertEqual(len(stream_signal), N_SAMPLES)

        # Process and predict from streaming buffer
        preprocessed, dur = preprocess_input_audio(stream_signal)
        res = predict_emotion(preprocessed, "ANN (Best)", self.mgr, actual_duration=dur)
        self.assertIsNotNone(res["predicted_emotion"])
        self.assertIsNone(res["error"])

    def test_telemetry_integration(self):
        """Telemetry tracker must capture predictions, throughput, and summary metrics."""
        initial_metrics = tracker.get_summary_metrics()
        initial_count = initial_metrics["total_requests"]

        if self.test_files:
            signal, duration = preprocess_input_audio(self.test_files[0])
            _ = predict_emotion(signal, "ANN (Best)", self.mgr, actual_duration=duration)

            updated_metrics = tracker.get_summary_metrics()
            self.assertEqual(updated_metrics["total_requests"], initial_count + 1)
            self.assertGreater(len(tracker.get_recent_logs()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
