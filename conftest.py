"""Pytest configuration: ensure the project root is importable and set a
deterministic Keras backend for tests."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, str(Path(__file__).resolve().parent))
