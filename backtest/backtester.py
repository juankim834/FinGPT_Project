"""
End-to-end news-to-signal backtester.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import pandas as pd

from agent1.extractor import extract_fingerprint
from agent2.reasoner import generate_signal
from backtest.dataset_parser import build_backtest_rows, load_dataset
from backtest.price_fetcher import direction_from_return, get_realized_return
from config import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def _position_from_direction(direction: str) -> int:
    if direction == "long":
        return 1
    if direction == "short":
        return -1
    return 0


def run_backtest(
    dataset_path: str,
    output_path: str = "output/backtest_results.csv",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the full news2signal pipeline and persist detailed results.
    """
    df = load_dataset(dataset_path)
    rows = build_backtest_rows(df)
    if max_rows is not None:
        rows = rows[:max_rows]

    results: list[dict] = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        if idx % 10 == 0:
            logger.info("Backtest progress: %d/%d rows processed", idx, total)

        base_result = {
            "ticker": row["ticker"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "article_text": row["article_text"][:120],
            "fingpt_label": row["fingpt_label"],
            "signal_direction": None,
            "signal_confidence": None,
            "signal_strategy_tag": None,
            "realized_return": None,
            "position": None,
            "strategy_return": None,
            "skipped_reason": "",
        }

        try:
            fingerprint = extract_fingerprint(row["article_text"])
            if fingerprint is None:
                base_result["skipped_reason"] = "fingerprint_failed"
                results.append(base_result)
                continue

            signal = generate_signal(fingerprint)
            if signal is None:
                base_result["skipped_reason"] = "signal_failed"
                results.append(base_result)
                continue

            realized_return = get_realized_return(
                ticker=row["ticker"],
                start_date=row["start_date"],
                end_date=row["end_date"],
            )
            if realized_return is None:
                base_result["signal_direction"] = signal.direction
                base_result["signal_confidence"] = signal.confidence
                base_result["signal_strategy_tag"] = signal.strategy_tag
                base_result["skipped_reason"] = "price_fetch_failed"
                results.append(base_result)
                continue

            position = _position_from_direction(signal.direction)
            strategy_return = position * realized_return

            base_result["signal_direction"] = signal.direction
            base_result["signal_confidence"] = signal.confidence
            base_result["signal_strategy_tag"] = signal.strategy_tag
            base_result["realized_return"] = realized_return
            base_result["position"] = position
            base_result["strategy_return"] = strategy_return
            results.append(base_result)
        except Exception as exc:
            logger.exception("Failed to process backtest row %d: %s", idx, exc)
            if not base_result["skipped_reason"]:
                base_result["skipped_reason"] = "signal_failed"
            results.append(base_result)

    results_df = pd.DataFrame(results)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Saved backtest results to %s", output_path)
    return results_df


def compute_metrics(results: pd.DataFrame) -> dict:
    """
    Compute backtest summary metrics over successful rows only.
    """
    total_rows = int(len(results))
    successful = results[results["skipped_reason"] == ""].copy()
    successful_rows = int(len(successful))
    skipped = total_rows - successful_rows
    skip_rate = (skipped / total_rows) if total_rows else 0.0

    metrics = {
        "total_rows": total_rows,
        "successful_rows": successful_rows,
        "skip_rate": skip_rate,
        "direction_accuracy": 0.0,
        "long_accuracy": 0.0,
        "short_accuracy": 0.0,
        "mean_strategy_return": 0.0,
        "std_strategy_return": 0.0,
        "annualized_sharpe": 0.0,
        "total_pnl": 0.0,
        "vs_fingpt_accuracy": 0.0,
    }
    if successful_rows == 0:
        return metrics

    successful["realized_direction"] = successful["realized_return"].astype(float).apply(
        direction_from_return
    )
    successful["signal_market_direction"] = successful["signal_direction"].map(
        {"long": "up", "short": "down", "neutral": "neutral"}
    )
    successful["is_direction_correct"] = (
        successful["signal_market_direction"] == successful["realized_direction"]
    )
    metrics["direction_accuracy"] = float(successful["is_direction_correct"].mean())

    long_rows = successful[successful["signal_direction"] == "long"]
    if not long_rows.empty:
        metrics["long_accuracy"] = float(
            (long_rows["realized_direction"] == "up").mean()
        )

    short_rows = successful[successful["signal_direction"] == "short"]
    if not short_rows.empty:
        metrics["short_accuracy"] = float(
            (short_rows["realized_direction"] == "down").mean()
        )

    strategy_returns = successful["strategy_return"].astype(float)
    mean_return = float(strategy_returns.mean())
    std_return = float(strategy_returns.std(ddof=0))
    metrics["mean_strategy_return"] = mean_return
    metrics["std_strategy_return"] = std_return
    metrics["total_pnl"] = float(strategy_returns.sum())
    if std_return > 0:
        metrics["annualized_sharpe"] = float((mean_return / std_return) * math.sqrt(52.0))

    successful["fingpt_direction"] = successful["fingpt_label"].map(
        {"up": "long", "down": "short", "neutral": "neutral"}
    )
    metrics["vs_fingpt_accuracy"] = float(
        (successful["signal_direction"] == successful["fingpt_direction"]).mean()
    )

    return metrics

