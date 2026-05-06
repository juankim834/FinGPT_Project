import pandas as pd
from pathlib import Path
from uuid import uuid4

from backtest.backtester import run_backtest


def test_run_backtest_respects_skip_llm_marker(monkeypatch):
    case_dir = Path("output/test_artifacts") / f"skip_llm_{uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = case_dir / "skip_llm_dataset.csv"
    output_path = case_dir / "results.csv"

    pd.DataFrame(
        [
            {
                "input": "",
                "output": "",
                "answer": "",
                "ticker": "AAPL",
                "skip_llm": True,
                "forced_signal": "no_signal",
                "skip_reason": "no_article_provided",
                "pass_reason": "no_article_provided",
            }
        ]
    ).to_csv(dataset_path, index=False)

    def _fail_extract(*args, **kwargs):
        raise AssertionError("LLM extraction should not be called for skip_llm rows")

    def _fail_signal(*args, **kwargs):
        raise AssertionError("LLM signal generation should not be called for skip_llm rows")

    monkeypatch.setattr("backtest.backtester.extract_fingerprint_batch", _fail_extract)
    monkeypatch.setattr("backtest.backtester.generate_signal_batch", _fail_signal)

    results = run_backtest(str(dataset_path), output_path=str(output_path))

    assert len(results) == 1
    row = results.iloc[0]
    assert row["direction"] == "no_signal"
    assert row["signal_direction"] == "no_signal"
    assert row["pass_reason"] == "no_article_provided"
    assert row["skipped_reason"] == "no_article_provided"
