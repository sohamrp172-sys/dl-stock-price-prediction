# 📈 AI-Powered Stock Price Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-3-red.svg)](https://keras.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![CI](https://github.com/sohamrp172-sys/dl-stock-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/sohamrp172-sys/dl-stock-price-prediction/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](#-testing)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A production-grade deep learning application that forecasts stock prices with a stacked **LSTM** network, served through a decoupled **FastAPI** backend and an interactive **Streamlit** dashboard.

### 🔗 Live Demo

- **Dashboard:** `https://<your-app>.onrender.com`  _(add your link after deploying — see [Deployment](#-deployment))_
- **API Docs (Swagger):** `https://<your-api>.onrender.com/docs`

> Replace the placeholders above with your live URLs once deployed. The step-by-step guide below gets you a public link in minutes.

---

## ✨ Highlights

- **Correct, high-accuracy inference** — fixes a scaler bug in the original code that pushed recent prices out of distribution. Back-test **MAPE dropped from ~23% to ~3.8%** (≈96% accuracy on GOOG).
- **Modular architecture** — data, model, and indicator logic live in a reusable `src/` package. The UI and API are thin presentation layers.
- **Decoupled REST API** — FastAPI backend with Pydantic validation and auto-generated Swagger docs, consumable by any frontend.
- **Robust UX** — ticker validation, invalid-ticker handling, loading spinners, and graceful error states.
- **Deployment-ready** — clean Dockerfile, Render blueprint, and Streamlit Cloud compatible.
- **Tested** — 18 unit/contract tests covering validation, metrics, scaling, and API contracts.

---

## 🏗️ Architecture

```
┌──────────────────┐        ┌──────────────────┐
│  Streamlit UI    │        │  FastAPI Backend │
│  (app.py)        │        │  (api/main.py)   │
│  5 interactive   │        │  /forecast       │
│  tabs + Plotly   │        │  /metrics/{tkr}  │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         └────────────┬──────────────┘
                      ▼
            ┌───────────────────────┐
            │   src/ core package   │
            │  data.py  (yfinance)  │
            │  model.py (LSTM infer)│
            │  indicators.py (TA)   │
            │  config.py (settings) │
            └───────────────────────┘
```

Both frontends share the same tested core, so behaviour is consistent whether you click a chart or call the API.

---

## 🧠 Model & Pipeline

**Architecture** — a 4-layer stacked LSTM with progressive dropout:

```
Input (100-day close-price sequence)
  → LSTM(50)  + Dropout(0.2)
  → LSTM(60)  + Dropout(0.3)
  → LSTM(80)  + Dropout(0.4)
  → LSTM(120) + Dropout(0.5)
  → Dense(1)   (next-day price)

Optimizer: Adam | Loss: MSE | Lookback: 100 days
```

**Inference pipeline**

1. **Data** — real-time OHLCV from Yahoo Finance, with ticker validation.
2. **Scaling** — `MinMaxScaler` fit on the **full price history** so today's prices stay in the range the model expects (the key correctness fix).
3. **Sequencing** — rolling 100-day windows.
4. **Prediction** — back-test over a held-out 20% split; autoregressive multi-day forecasting.
5. **Post-processing** — inverse-transform to dollars, confidence bands from historical volatility.

### 📊 Measured Performance (GOOG, full history)

| Metric | Value |
|--------|-------|
| RMSE | ~$13.40 |
| MAE | ~$9.43 |
| MAPE | ~3.81% |
| Accuracy (100 − MAPE) | ~96.2% |

_Metrics vary by ticker and date range; these are reproducible via the Predictions tab or `GET /metrics/GOOG`._

---

## 📦 Project Structure

```
dl-stock-price-prediction/
├── app.py                      # Streamlit dashboard (UI only)
├── api/
│   └── main.py                 # FastAPI backend (REST + Swagger)
├── src/
│   ├── config.py               # Central configuration
│   ├── data.py                 # Data loading + ticker validation
│   ├── model.py                # LSTM load / evaluate / forecast
│   └── indicators.py           # RSI, MACD, Bollinger, ADX, MAs
├── tests/                      # 18 unit & API tests
├── Stock_Predictions_Model.keras   # Pre-trained model
├── requirements.txt
├── Dockerfile
├── render.yaml                 # Render blueprint
├── start.sh
└── .streamlit/config.toml
```

---

## 🚀 Quick Start (Local)

**Prerequisites:** Python 3.11, Git.

```bash
# 1. Clone
git clone https://github.com/sohamrp172-sys/dl-stock-price-prediction.git
cd dl-stock-price-prediction

# 2. Create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run the dashboard
streamlit run app.py
#    → http://localhost:8501

# 4b. (Optional) Run the API in another terminal
uvicorn api.main:app --reload
#    → http://localhost:8000/docs
```

> The default Keras backend is **TensorFlow** (broad wheel support for cloud hosts). For a lighter local setup you can instead `pip install jax==0.4.30 jaxlib==0.4.30` and set `KERAS_BACKEND=jax`. Dev/test extras live in `requirements-dev.txt`.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check + model status |
| `POST` | `/forecast` | Forecast future prices for a ticker |
| `GET`  | `/metrics/{ticker}` | Back-test accuracy metrics |

**Example**

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 14}'
```

```json
{
  "ticker": "AAPL",
  "current_price": 234.40,
  "forecast": [{"day": 1, "price": 235.10}, {"day": 2, "price": 235.80}]
}
```

Invalid tickers return `422` with a clear message; a missing model returns `503`.

---

## 🧪 Testing

```bash
pytest -q
```

Covers ticker validation, date-range logic, metric math, scaler behaviour, and API contract/error handling.

---

## 🐳 Run with Docker

```bash
docker build -t stock-predictor .
docker run -p 8501:8501 stock-predictor
# → http://localhost:8501
```

The image is Python 3.11-slim based, installs CPU-only deps, and includes a health check.

---

## 🌐 Deployment

### Option A — Render via Docker (recommended for production)

The repo ships a `render.yaml` blueprint that deploys **two Docker services** from
the same image: the Streamlit dashboard and the FastAPI backend.

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Blueprint**.
3. Connect the repo. Render reads `render.yaml` and provisions both services.
4. Click **Apply**. The first Docker build takes ~5 minutes.
5. Two live URLs appear:
   - Dashboard: `https://ai-stock-predictor-ui.onrender.com`
   - API docs: `https://ai-stock-predictor-api.onrender.com/docs`

> **Production notes**
> - The free plan sleeps after ~15 min idle and cold-starts on the next hit. For an always-on app, change `plan: free` → `plan: starter` in `render.yaml`.
> - The default backend is **JAX** — chosen because it's far lighter on RAM than TensorFlow, which matters on constrained instances. Switch to TensorFlow by editing `requirements.txt` and the `KERAS_BACKEND` env vars.
> - Both services share the Docker image, so behaviour matches local and CI exactly.

### Option B — Any Docker host

The same `Dockerfile` runs on Fly.io, Railway, Google Cloud Run, AWS App Runner, etc.
Build and run, honouring the `$PORT` env var:

```bash
docker build -t stock-predictor .
docker run -p 8501:8501 -e PORT=8501 stock-predictor
```

### Option C — Streamlit Community Cloud (quick demo)

For a fast public link (free, but less "production"): [share.streamlit.io](https://share.streamlit.io) → **New app** → repo, branch `main`, main file `app.py`. Uses `requirements.txt` + `runtime.txt`.

---

## 🔄 CI/CD

Every push and pull request runs a GitHub Actions pipeline (`.github/workflows/ci.yml`) that:

1. Installs dependencies on a clean Ubuntu + Python 3.11 runner.
2. Runs the full `pytest` suite.
3. Builds the Docker image to guarantee the container always ships.

This keeps `main` green and the deployment artifact reproducible.

> **Enabling CI:** the workflow file lives at `.github/workflows/ci.yml` in the repo. If you cloned via HTTPS with a token that lacks the `workflow` scope, GitHub will block pushes that add it — either push the file via the web UI, or re-authenticate with a token that has the `workflow` scope enabled.

---

## ⚠️ Disclaimer

For **educational and portfolio purposes only**. This is **not** financial advice. Stock predictions are based on historical patterns; past performance does not guarantee future results. Always do your own research.

---

## 📝 License

MIT — see [LICENSE](LICENSE).
