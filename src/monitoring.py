"""
src/monitoring.py
─────────────────
Production monitoring, telemetry, and system health tracking for SpeakSense SER.
Tracks real-time prediction latency, model throughput, emotion distributions,
system resources, and provides auditable session logs.
"""

from __future__ import annotations

import datetime
import io
import os
import platform
import sys
import threading
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class TelemetryTracker:
    """Thread-safe in-memory session telemetry and performance tracker."""

    _instance: Optional[TelemetryTracker] = None
    _lock = threading.Lock()

    def __new__(cls) -> TelemetryTracker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_tracker()
            return cls._instance

    def _init_tracker(self) -> None:
        self._start_time = time.time()
        self._logs: deque = deque(maxlen=500)   # keep last 500 predictions
        self._latencies: deque = deque(maxlen=500)
        self._emotion_counter: Counter = Counter()
        self._model_counter: Counter = Counter()
        self._error_counter: Counter = Counter()
        self._total_requests = 0
        self._tracker_lock = threading.Lock()

    def record_prediction(
        self,
        model_name: str,
        predicted_emotion: str,
        confidence: float,
        latency_ms: float,
        audio_duration_sec: float,
        input_type: str = "upload",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an inference event into session telemetry."""
        record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "predicted_emotion": predicted_emotion,
            "confidence": round(float(confidence), 2) if confidence is not None else 0.0,
            "latency_ms": round(float(latency_ms), 2) if latency_ms is not None else 0.0,
            "audio_duration_sec": round(float(audio_duration_sec), 2) if audio_duration_sec is not None else 0.0,
            "input_type": input_type,
            "status": "ERROR" if error else "SUCCESS",
            "error_msg": str(error) if error else "",
        }

        with self._tracker_lock:
            self._total_requests += 1
            self._logs.append(record)
            if error:
                self._error_counter[str(error)] += 1
            else:
                self._latencies.append(latency_ms)
                self._emotion_counter[predicted_emotion] += 1
                self._model_counter[model_name] += 1

        return record

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute aggregated latency and throughput metrics."""
        with self._tracker_lock:
            total = self._total_requests
            latencies = list(self._latencies)
            errors = sum(self._error_counter.values())

        if latencies:
            arr = np.array(latencies, dtype=np.float32)
            avg_lat = float(np.mean(arr))
            p50_lat = float(np.percentile(arr, 50))
            p95_lat = float(np.percentile(arr, 95))
            min_lat = float(np.min(arr))
            max_lat = float(np.max(arr))
        else:
            avg_lat = p50_lat = p95_lat = min_lat = max_lat = 0.0

        uptime_sec = time.time() - self._start_time

        return {
            "total_requests": total,
            "successful_requests": total - errors,
            "error_count": errors,
            "error_rate_pct": round((errors / total * 100) if total > 0 else 0.0, 1),
            "uptime_formatted": str(datetime.timedelta(seconds=int(uptime_sec))),
            "avg_latency_ms": round(avg_lat, 2),
            "p50_latency_ms": round(p50_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "emotion_distribution": dict(self._emotion_counter),
            "model_distribution": dict(self._model_counter),
        }

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent prediction logs in reverse chronological order."""
        with self._tracker_lock:
            logs = list(self._logs)
        return logs[-limit:][::-1]

    def export_csv(self) -> str:
        """Export all session logs to CSV formatted string."""
        with self._tracker_lock:
            logs = list(self._logs)
        if not logs:
            return "timestamp,model_name,predicted_emotion,confidence,latency_ms,audio_duration_sec,input_type,status,error_msg\n"
        df = pd.DataFrame(logs)
        return df.to_csv(index=False)

    def get_system_health(self) -> Dict[str, Any]:
        """Query host environment and resource utilization."""
        health = {
            "python_version": platform.python_version(),
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "cpu_cores": os.cpu_count() or 1,
            "memory_usage_mb": 0.0,
            "status": "HEALTHY",
        }

        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            health["memory_usage_mb"] = round(mem_info.rss / (1024 * 1024), 1)
            health["cpu_percent"] = psutil.cpu_percent(interval=None)
        except ImportError:
            # Fallback if psutil is not installed
            health["memory_usage_mb"] = "N/A (psutil optional)"
            health["cpu_percent"] = "N/A"

        return health

    def reset(self) -> None:
        """Reset telemetry session."""
        with self._tracker_lock:
            self._init_tracker()


# Global singleton helper
tracker = TelemetryTracker()
