#!/bin/bash
# Start script for cloud deployment (Render / generic container hosts)
set -e

echo "Starting AI-Powered Stock Price Predictor..."

export KERAS_BACKEND=${KERAS_BACKEND:-tensorflow}
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

streamlit run app.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true
