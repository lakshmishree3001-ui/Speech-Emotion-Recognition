"""
tests/test_deployment.py
────────────────────────
Module 14 Automated Test Suite: Deployment Readiness, Preflight Checks & Live Health.
Verifies all configuration files, models, manifests, packages.txt, requirements.txt,
and performs HTTP health checks against local or deployed Streamlit endpoints.
"""

import argparse
import hashlib
import json
import os
import sys
import unittest

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predictor import ModelManager


class TestDeploymentReadiness(unittest.TestCase):
    """Preflight and integrity validation for cloud deployment (Module 14)."""

    def test_essential_files_exist(self):
        """Verify all critical deployment configuration files exist."""
        required_files = [
            "app/app.py",
            "src/predictor.py",
            "src/audio_preprocessing.py",
            "src/monitoring.py",
            "requirements.txt",
            "packages.txt",
            ".streamlit/config.toml",
            "Dockerfile",
            ".dockerignore",
            "DEPLOYMENT.md",
            "models/model_manifest.json",
            "models/preprocessing_config.json",
            "models/scaler.pkl",
            "models/label_encoder.pkl",
        ]
        for rel_path in required_files:
            full_path = os.path.join(BASE_DIR, rel_path)
            self.assertTrue(
                os.path.exists(full_path),
                f"Missing critical deployment file: {rel_path}"
            )

    def test_packages_txt_content(self):
        """packages.txt must contain libsndfile1 and ffmpeg for Linux audio decoding."""
        pkg_path = os.path.join(BASE_DIR, "packages.txt")
        with open(pkg_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
        self.assertIn("libsndfile1", content)
        self.assertIn("ffmpeg", content)

    def test_requirements_txt_content(self):
        """requirements.txt must contain streamlit, librosa, soundfile, and tensorflow."""
        req_path = os.path.join(BASE_DIR, "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
        required_pkgs = ["streamlit", "librosa", "soundfile", "scikit-learn", "tensorflow", "psutil"]
        for pkg in required_pkgs:
            self.assertIn(pkg, content, f"Missing {pkg} in requirements.txt")

    def test_model_manifest_integrity(self):
        """Every model listed in model_manifest.json must exist and match its recorded SHA-256 hash."""
        manifest_path = os.path.join(BASE_DIR, "models", "model_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertGreaterEqual(len(manifest), 5)

        for model_name, info in manifest.items():
            model_file = os.path.join(BASE_DIR, "models", info["filename"])
            self.assertTrue(os.path.exists(model_file), f"Model file missing: {model_file}")

            # Verify sha256 hash
            hasher = hashlib.sha256()
            with open(model_file, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            current_hash = hasher.hexdigest()
            self.assertEqual(
                current_hash, info["sha256"],
                f"Hash mismatch for {model_name}! Expected {info['sha256']}, got {current_hash}"
            )

    def test_model_manager_readiness(self):
        """ModelManager must report healthy and load without errors."""
        mgr = ModelManager()
        status = mgr.load_all()
        health = mgr.get_health_status()
        self.assertTrue(health["healthy"])
        self.assertGreaterEqual(health["total_models_loaded"], 5)
        self.assertTrue(health["manifest_present"])

    def test_gitignore_protects_raw_data(self):
        """.gitignore must exclude Data/Raw/ to prevent GitHub file limit issues."""
        gi_path = os.path.join(BASE_DIR, ".gitignore")
        with open(gi_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Data/Raw/", content)


def check_deployed_endpoint(url: str) -> bool:
    """Optional utility to verify a deployed Streamlit endpoint over HTTP."""
    try:
        import urllib.request
        health_url = url.rstrip("/") + "/_stcore/health"
        req = urllib.request.Request(
            health_url,
            headers={"User-Agent": "SpeakSense-Deployment-Tester/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            print(f"Health check to {health_url}: HTTP {status}")
            return status == 200
    except Exception as e:
        print(f"Endpoint health check failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test deployment readiness and optional live endpoint.")
    parser.add_argument("--url", type=str, default=None, help="Deployed Streamlit app URL to check health.")
    args, unknown = parser.parse_known_args()

    if args.url:
        print(f"Checking deployed URL: {args.url}")
        success = check_deployed_endpoint(args.url)
        sys.exit(0 if success else 1)

    # Run standard test suite
    unittest.main(argv=[sys.argv[0]] + unknown, verbosity=2)
