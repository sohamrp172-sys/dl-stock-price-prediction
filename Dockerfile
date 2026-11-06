# ──────────────────────────────────────────────────────────────
# AI-Powered Stock Price Predictor — production image
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=jax \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps occasionally needed by scientific wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

EXPOSE 8501

# Note: health checks are handled by the hosting platform via an HTTP path
# (see `healthCheckPath` in render.yaml). We intentionally do not hardcode a
# container HEALTHCHECK here because the same image serves two different
# processes (Streamlit on /_stcore/health and FastAPI on /).

# Default: run the Streamlit dashboard. Honour $PORT for cloud platforms.
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
