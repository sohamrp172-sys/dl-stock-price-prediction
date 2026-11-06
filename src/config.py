"""Central configuration for the stock prediction project.

All tunable constants live here so the training, inference, API, and UI layers
share a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "Stock_Predictions_Model.keras"))

# ---------------------------------------------------------------------------
# Keras / backend configuration
# ---------------------------------------------------------------------------
# TensorFlow is the default backend: it has broad, reliable wheel support on
# cloud hosts (Streamlit Cloud, Render). Override via KERAS_BACKEND=jax for
# lightweight local development if preferred.
KERAS_BACKEND = os.getenv("KERAS_BACKEND", "tensorflow")

# ---------------------------------------------------------------------------
# Model / data hyper-parameters
# ---------------------------------------------------------------------------
LOOKBACK_WINDOW = 100          # days of history fed into the LSTM
TRAIN_SPLIT = 0.80             # fraction of data used for training
DEFAULT_START_DATE = "2012-06-01"
MAX_FUTURE_DAYS = 90
MIN_FUTURE_DAYS = 7
DEFAULT_FUTURE_DAYS = 30

# ---------------------------------------------------------------------------
# Networking (FastAPI)
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
DATA_CACHE_TTL = 3600          # seconds
