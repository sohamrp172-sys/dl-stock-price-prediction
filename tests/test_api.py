"""API contract tests using FastAPI's TestClient.

Only the health endpoint is exercised offline; forecast/metrics require network
and the model, so they are smoke-checked for correct error handling on bad input.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "jax")

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_forecast_rejects_bad_ticker():
    resp = client.post("/forecast", json={"ticker": "A@PL", "days": 10})
    assert resp.status_code == 422


def test_forecast_rejects_out_of_range_days():
    # days below the allowed minimum should fail Pydantic validation.
    resp = client.post("/forecast", json={"ticker": "AAPL", "days": 1})
    assert resp.status_code == 422
