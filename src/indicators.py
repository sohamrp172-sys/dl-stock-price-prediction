"""Technical indicator calculations.

Thin wrappers around the `ta` library so the UI/API never touch it directly.
Each function is defensive: it returns the enriched DataFrame and never raises
on short series (indicators simply contain NaNs until enough data accrues).
"""
from __future__ import annotations

import pandas as pd
import ta


def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """Add 50/100/200-day simple moving averages of the close price."""
    data = data.copy()
    data["MA_50"] = data["Close"].rolling(window=50).mean()
    data["MA_100"] = data["Close"].rolling(window=100).mean()
    data["MA_200"] = data["Close"].rolling(window=200).mean()
    return data


def add_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    data = data.copy()
    data["RSI"] = ta.momentum.rsi(data["Close"], window=window)
    return data


def add_macd(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    macd = ta.trend.MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Hist"] = macd.macd_diff()
    return data


def bollinger_position(data: pd.DataFrame) -> float:
    """Return the latest price position within the Bollinger Bands (0-100%)."""
    bb = ta.volatility.BollingerBands(data["Close"])
    high = bb.bollinger_hband().iloc[-1]
    low = bb.bollinger_lband().iloc[-1]
    if high == low:
        return 50.0
    return float((data["Close"].iloc[-1] - low) / (high - low) * 100)


def adx_strength(data: pd.DataFrame) -> float:
    adx = ta.trend.ADXIndicator(data["High"], data["Low"], data["Close"])
    return float(adx.adx().iloc[-1])


def enrich(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the full indicator suite used by the dashboard."""
    data = add_moving_averages(data)
    data = add_rsi(data)
    data = add_macd(data)
    return data
