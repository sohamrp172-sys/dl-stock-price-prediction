"""Data acquisition and validation layer.

Wraps yfinance with input validation, graceful error handling for invalid
tickers, and normalisation of the DataFrame shape (yfinance returns a
MultiIndex for single tickers in recent versions).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .config import DEFAULT_START_DATE

# A ticker is 1-10 chars: letters, digits, dots and dashes (e.g. BRK.B, 005930.KS)
_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class InvalidTickerError(ValueError):
    """Raised when a ticker symbol is malformed or returns no data."""


def validate_ticker(ticker: str) -> str:
    """Validate and normalise a ticker symbol.

    Raises:
        InvalidTickerError: if the symbol is empty or malformed.
    """
    if not ticker or not ticker.strip():
        raise InvalidTickerError("Ticker symbol cannot be empty.")
    normalised = ticker.strip().upper()
    if not _TICKER_RE.match(normalised):
        raise InvalidTickerError(
            f"'{ticker}' is not a valid ticker. Use letters, digits, '.' or '-' "
            "(e.g. AAPL, MSFT, BRK.B)."
        )
    return normalised


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns to simple OHLCV names."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def load_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download historical OHLCV data for a ticker.

    Args:
        ticker: Stock symbol (validated and normalised internally).
        start_date: ISO date string. Defaults to project start date.
        end_date: ISO date string. Defaults to today.

    Returns:
        A DataFrame indexed by date with OHLCV columns.

    Raises:
        InvalidTickerError: if the ticker is malformed or returns no data.
    """
    symbol = validate_ticker(ticker)
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")

    try:
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:  # network / yfinance internal errors
        raise InvalidTickerError(
            f"Failed to download data for '{symbol}': {exc}"
        ) from exc

    if data is None or data.empty:
        raise InvalidTickerError(
            f"No data found for ticker '{symbol}'. It may be delisted or incorrect."
        )

    data = _flatten_columns(data)

    missing = [c for c in OHLCV_COLUMNS if c not in data.columns]
    if missing:
        raise InvalidTickerError(
            f"Data for '{symbol}' is missing columns: {missing}."
        )

    return data.dropna(subset=["Close"])


def date_range_for_period(period: str) -> tuple[str, str]:
    """Translate a human period label into (start, end) ISO date strings."""
    end = datetime.now()
    lookup = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "5 Years": 1825,
    }
    if period in lookup:
        start = end - timedelta(days=lookup[period])
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    # "Max" or unknown -> project default
    return DEFAULT_START_DATE, end.strftime("%Y-%m-%d")
