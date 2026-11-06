"""FastAPI backend exposing the prediction pipeline as a REST service.

This decoupled API lets the project be consumed by any frontend (the Streamlit
dashboard, a mobile client, or an automated trading research script) and makes
the model's capabilities demonstrable via interactive Swagger docs at /docs.
"""
from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import __version__
from src.config import DEFAULT_FUTURE_DAYS, MAX_FUTURE_DAYS, MIN_FUTURE_DAYS
from src.data import InvalidTickerError, load_stock_data
from src.model import evaluate, forecast_future, load_prediction_model

app = FastAPI(
    title="AI Stock Price Predictor API",
    description="LSTM-based stock price forecasting and back-testing service.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    version: str
    model_loaded: bool


class ForecastRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], description="Stock ticker symbol")
    days: int = Field(
        DEFAULT_FUTURE_DAYS,
        ge=MIN_FUTURE_DAYS,
        le=MAX_FUTURE_DAYS,
        description="Number of future days to forecast",
    )
    start_date: str | None = Field(None, description="ISO start date (optional)")


class ForecastPoint(BaseModel):
    day: int
    price: float


class ForecastResponse(BaseModel):
    ticker: str
    current_price: float
    forecast: List[ForecastPoint]


class MetricsResponse(BaseModel):
    ticker: str
    rmse: float
    mae: float
    mape: float
    accuracy: float
    test_points: int


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check."""
    loaded = True
    try:
        load_prediction_model()
    except Exception:
        loaded = False
    return HealthResponse(status="ok", version=__version__, model_loaded=loaded)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    """Forecast future closing prices for a ticker."""
    try:
        data = load_stock_data(req.ticker, start_date=req.start_date)
    except InvalidTickerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        model = load_prediction_model()
        preds = forecast_future(data, req.days, model=model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc

    return ForecastResponse(
        ticker=req.ticker.upper(),
        current_price=float(data["Close"].iloc[-1]),
        forecast=[ForecastPoint(day=i + 1, price=round(float(p), 2)) for i, p in enumerate(preds)],
    )


@app.get("/metrics/{ticker}", response_model=MetricsResponse)
def metrics(ticker: str) -> MetricsResponse:
    """Back-test the model on a ticker and return accuracy metrics."""
    try:
        data = load_stock_data(ticker)
    except InvalidTickerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = evaluate(data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return MetricsResponse(
        ticker=ticker.upper(),
        rmse=round(result.rmse, 2),
        mae=round(result.mae, 2),
        mape=round(result.mape, 2),
        accuracy=round(result.accuracy, 2),
        test_points=len(result.y_true),
    )
