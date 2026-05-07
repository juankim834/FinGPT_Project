import math

import pandas as pd

from backtest.price_fetcher import _fetch_yfinance_return


def test_fetch_yfinance_return_treats_end_date_as_inclusive(monkeypatch):
    captured: dict[str, str] = {}

    def _fake_download(*, tickers, start, end, interval, auto_adjust, progress, threads):
        captured["tickers"] = tickers
        captured["start"] = start
        captured["end"] = end
        return pd.DataFrame({"Close": [100.0, 110.0]})

    monkeypatch.setattr("backtest.price_fetcher.yf.download", _fake_download)

    realized_return, reason = _fetch_yfinance_return(
        ticker="AAPL",
        start_date="2024-01-08",
        end_date="2024-01-16",
    )

    assert reason == ""
    assert math.isclose(float(realized_return), 0.1)
    assert captured == {
        "tickers": "AAPL",
        "start": "2024-01-08",
        "end": "2024-01-17",
    }
