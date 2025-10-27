#!/bin/bash

# Start script for Render deployment

echo "Starting Stock Price Prediction App..."

# Set environment variables
export KERAS_BACKEND=jax
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit app
streamlit run app_enhanced.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false
