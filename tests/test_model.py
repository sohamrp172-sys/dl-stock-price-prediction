"""Unit tests for model helpers using synthetic data (no network/model load)."""
import numpy as np
import pandas as pd
import pytest

from src.model import _calculate_metrics, _fit_scaler, forecast_confidence


def test_calculate_metrics_perfect_prediction():
    actual = np.array([10.0, 20.0, 30.0])
    rmse, mae, mape, accuracy = _calculate_metrics(actual, actual.copy())
    assert rmse == pytest.approx(0.0)
    assert mae == pytest.approx(0.0)
    assert accuracy == pytest.approx(100.0)


def test_calculate_metrics_handles_zero_actual():
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([0.0, 11.0, 19.0])
    # Should not raise despite the zero entry.
    rmse, mae, mape, accuracy = _calculate_metrics(actual, predicted)
    assert mae > 0


def test_fit_scaler_range():
    prices = pd.Series(np.linspace(100, 200, 500))
    scaler = _fit_scaler(prices)
    transformed = scaler.transform(prices.values.reshape(-1, 1))
    assert transformed.min() >= 0.0
    # Values beyond the training slice may exceed 1 — that's expected.


def test_forecast_confidence_bounds():
    preds = np.array([100.0, 101.0, 102.0])
    assert forecast_confidence(preds, 0.0) == pytest.approx(100.0)
    assert forecast_confidence(preds, 1.0) == pytest.approx(50.0)
