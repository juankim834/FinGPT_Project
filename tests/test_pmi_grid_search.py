import math

import pandas as pd

from backtest.pmi_grid_search import (
    apply_pmi_alpha_to_results,
    parse_alpha_grid,
    run_pmi_alpha_grid_search,
)


def _base_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "raw_signal_logprob_A": 3.0,
                "raw_signal_logprob_B": 2.0,
                "raw_signal_logprob_C": 1.0,
                "pmi_null_logprob_A": 2.5,
                "pmi_null_logprob_B": 0.0,
                "pmi_null_logprob_C": 0.0,
                "realized_return": 0.1,
                "signal_direction": "long",
                "direction": "long",
                "strategy_return": 0.1,
                "skipped_reason": "",
            },
            {
                "ticker": "MSFT",
                "raw_signal_logprob_A": 1.0,
                "raw_signal_logprob_B": 0.5,
                "raw_signal_logprob_C": 2.5,
                "pmi_null_logprob_A": 0.0,
                "pmi_null_logprob_B": 0.0,
                "pmi_null_logprob_C": 0.0,
                "realized_return": -0.2,
                "signal_direction": "short",
                "direction": "short",
                "strategy_return": 0.2,
                "skipped_reason": "",
            },
        ]
    )


def test_apply_pmi_alpha_to_results_recomputes_direction_and_returns():
    results = _base_results()

    alpha_zero = apply_pmi_alpha_to_results(results, pmi_alpha=0.0, calibration_t=1.0)
    alpha_one = apply_pmi_alpha_to_results(results, pmi_alpha=1.0, calibration_t=1.0)

    assert alpha_zero.loc[0, "signal_direction"] == "long"
    assert alpha_zero.loc[0, "position"] == 1
    assert math.isclose(float(alpha_zero.loc[0, "strategy_return"]), 0.1)

    assert alpha_one.loc[0, "signal_direction"] == "neutral"
    assert alpha_one.loc[0, "position"] == 0
    assert math.isclose(float(alpha_one.loc[0, "strategy_return"]), 0.0)
    assert alpha_one.loc[0, "signal_filter_forced_hold"] in {False, True}


def test_run_pmi_alpha_grid_search_returns_summary_metrics():
    summary = run_pmi_alpha_grid_search(
        _base_results(),
        alphas=[0.0, 1.0],
        calibration_t=1.0,
    )

    assert list(summary["pmi_alpha"]) == [0.0, 1.0]
    assert "total_pnl" in summary.columns
    assert "annualized_sharpe" in summary.columns
    assert "rank_total_pnl" in summary.columns
    assert "signal_changed_count" in summary.columns
    assert "signal_long_count" in summary.columns
    assert float(summary.loc[summary["pmi_alpha"] == 0.0, "total_pnl"].iloc[0]) > float(
        summary.loc[summary["pmi_alpha"] == 1.0, "total_pnl"].iloc[0]
    )


def test_parse_alpha_grid_handles_comma_separated_list():
    assert parse_alpha_grid("0, 0.25,1.0") == [0.0, 0.25, 1.0]
