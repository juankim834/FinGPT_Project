import math

import pandas as pd

from backtest.backtester import reprice_backtest_results


def _make_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "start_date": "2024-01-01",
                "end_date": "2024-01-08",
                "signal_direction": "long",
                "direction": "long",
                "realized_return": None,
                "position": None,
                "strategy_return": None,
                "skipped_reason": "price_fetch_failed",
            },
            {
                "ticker": "MSFT",
                "start_date": "2024-01-01",
                "end_date": "2024-01-08",
                "signal_direction": "short",
                "direction": "short",
                "realized_return": None,
                "position": None,
                "strategy_return": None,
                "skipped_reason": "price_fetch_failed",
            },
            {
                "ticker": "TSLA",
                "start_date": "2024-01-01",
                "end_date": "2024-01-08",
                "signal_direction": None,
                "direction": None,
                "realized_return": None,
                "position": None,
                "strategy_return": None,
                "skipped_reason": "fingerprint_failed",
            },
        ]
    )


def test_reprice_backtest_results_recovers_price_failed_rows(monkeypatch):
    calls: list[tuple[str, bool]] = []

    def _fake_get_realized_return_with_reason(ticker, start_date, end_date, *, refresh):
        calls.append((ticker, refresh))
        if ticker == "AAPL":
            return 0.1, ""
        return None, "empty_history"

    monkeypatch.setattr(
        "backtest.backtester.get_realized_return_with_reason",
        _fake_get_realized_return_with_reason,
    )

    repriced = reprice_backtest_results(_make_results_df(), refresh_prices=True)

    assert calls == [("AAPL", True), ("MSFT", True)]

    aapl = repriced.iloc[0]
    assert aapl["skipped_reason"] == ""
    assert aapl["position"] == 1
    assert math.isclose(float(aapl["realized_return"]), 0.1)
    assert math.isclose(float(aapl["strategy_return"]), 0.1)
    assert aapl["price_fetch_error_reason"] == ""

    msft = repriced.iloc[1]
    assert msft["skipped_reason"] == "price_fetch_failed"
    assert msft["price_fetch_error_reason"] == "empty_history"
    assert pd.isna(msft["realized_return"])
    assert pd.isna(msft["position"])
    assert pd.isna(msft["strategy_return"])

    tsla = repriced.iloc[2]
    assert tsla["skipped_reason"] == "fingerprint_failed"
    assert tsla["price_fetch_error_reason"] == ""


def test_reprice_backtest_results_preserves_existing_skip_reasons(monkeypatch):
    def _fake_get_realized_return_with_reason(ticker, start_date, end_date, *, refresh):
        return None, "missing_inputs"

    monkeypatch.setattr(
        "backtest.backtester.get_realized_return_with_reason",
        _fake_get_realized_return_with_reason,
    )

    df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "start_date": "",
                "end_date": "",
                "signal_direction": "long",
                "direction": "long",
                "skipped_reason": "signal_failed",
            }
        ]
    )

    repriced = reprice_backtest_results(df, refresh_prices=False)
    row = repriced.iloc[0]
    assert row["skipped_reason"] == "signal_failed"
    assert row["price_fetch_error_reason"] == "missing_inputs"
