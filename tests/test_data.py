"""Unit tests for the data validation layer (no network required)."""
import pytest

from src.data import InvalidTickerError, date_range_for_period, validate_ticker


@pytest.mark.parametrize("raw,expected", [
    ("aapl", "AAPL"),
    ("  msft  ", "MSFT"),
    ("BRK.B", "BRK.B"),
    ("005930.KS", "005930.KS"),
])
def test_validate_ticker_normalises(raw, expected):
    assert validate_ticker(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "TOOLONGTICKER123", "AA PL", "A@PL"])
def test_validate_ticker_rejects_bad_input(bad):
    with pytest.raises(InvalidTickerError):
        validate_ticker(bad)


def test_date_range_known_period():
    start, end = date_range_for_period("1 Month")
    assert start < end


def test_date_range_max_uses_default_start():
    start, _ = date_range_for_period("Max")
    assert start.startswith("2012")
