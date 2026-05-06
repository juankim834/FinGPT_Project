"""
End-to-end news-to-signal backtester.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import pandas as pd

from agent1.extractor import extract_fingerprint_batch
from agent2.reasoner import generate_signal_batch
from backtest.dataset_parser import build_backtest_rows, load_dataset
from backtest.price_fetcher import direction_from_return, get_realized_return
from config import (
    FINGPT_BACKTEST_BATCH_SIZE,
    FINGPT_BACKTEST_STRICT_MODE,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def _position_from_direction(direction: str) -> int:
    if direction == "long":
        return 1
    if direction == "short":
        return -1
    return 0


def _make_base_result(row: dict) -> dict:
    nan = math.nan
    return {
        "ticker": row["ticker"],
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "article_text": row["article_text"][:120],
        "fingpt_label": row["fingpt_label"],
        "sentiment_label": None,
        "sentiment_score": nan,
        "sentiment_confidence": nan,
        "sentiment_logprob_POSITIVE": nan,
        "sentiment_logprob_NEGATIVE": nan,
        "sentiment_logprob_NEUTRAL": nan,
        "sentiment_prob_POSITIVE": nan,
        "sentiment_prob_NEGATIVE": nan,
        "sentiment_prob_NEUTRAL": nan,
        "event_type": None,
        "event_type_confidence": nan,
        "event_type_margin": nan,
        "event_type_method": None,
        "event_logprob_A": nan,
        "event_logprob_B": nan,
        "event_logprob_C": nan,
        "event_logprob_D": nan,
        "event_logprob_E": nan,
        "event_logprob_F": nan,
        "event_logprob_G": nan,
        "event_prob_A": nan,
        "event_prob_B": nan,
        "event_prob_C": nan,
        "event_prob_D": nan,
        "event_prob_E": nan,
        "event_prob_F": nan,
        "event_prob_G": nan,
        "direction": None,
        "confidence": nan,
        "raw_signal_logprob_A": nan,
        "raw_signal_logprob_B": nan,
        "raw_signal_logprob_C": nan,
        "pmi_null_logprob_A": nan,
        "pmi_null_logprob_B": nan,
        "pmi_null_logprob_C": nan,
        "pmi_adjusted_logit_A": nan,
        "pmi_adjusted_logit_B": nan,
        "pmi_adjusted_logit_C": nan,
        "signal_prob_A": nan,
        "signal_prob_B": nan,
        "signal_prob_C": nan,
        "pmi_alpha_used": nan,
        "calibration_T": nan,
        "signal_filter_forced_hold": None,
        "signal_filter_reason": None,
        # Legacy aliases preserved for current metrics/downstream code.
        "signal_direction": None,
        "signal_confidence": nan,
        "strategy_tag": None,
        "realized_return": None,
        "position": None,
        "strategy_return": None,
        "skipped_reason": "",
    }


def _safe_list_item(values: object, index: int) -> float:
    if isinstance(values, list) and index < len(values):
        value = values[index]
        if isinstance(value, (int, float)):
            return float(value)
    return math.nan


def _safe_dict_item(values: object, key: str) -> float:
    if isinstance(values, dict):
        value = values.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return math.nan


def flatten_article_result(
    row: dict,
    fingerprint=None,
    signal=None,
) -> dict:
    result = _make_base_result(row)

    if fingerprint is not None:
        result["sentiment_label"] = fingerprint.sentiment_label
        result["sentiment_score"] = float(fingerprint.sentiment_score)
        result["sentiment_confidence"] = float(fingerprint.sentiment_confidence)
        result["sentiment_logprob_POSITIVE"] = _safe_list_item(
            fingerprint.sentiment_logits, 0
        )
        result["sentiment_logprob_NEGATIVE"] = _safe_list_item(
            fingerprint.sentiment_logits, 1
        )
        result["sentiment_logprob_NEUTRAL"] = _safe_list_item(
            fingerprint.sentiment_logits, 2
        )
        result["sentiment_prob_POSITIVE"] = _safe_dict_item(
            fingerprint.sentiment_probabilities, "POSITIVE"
        )
        result["sentiment_prob_NEGATIVE"] = _safe_dict_item(
            fingerprint.sentiment_probabilities, "NEGATIVE"
        )
        result["sentiment_prob_NEUTRAL"] = _safe_dict_item(
            fingerprint.sentiment_probabilities, "NEUTRAL"
        )
        result["event_type"] = fingerprint.event_type
        result["event_type_confidence"] = (
            float(fingerprint.event_type_confidence)
            if fingerprint.event_type_confidence is not None
            else math.nan
        )
        result["event_type_margin"] = (
            float(fingerprint.event_type_margin)
            if fingerprint.event_type_margin is not None
            else math.nan
        )
        result["event_type_method"] = fingerprint.event_type_method
        for token in ["A", "B", "C", "D", "E", "F", "G"]:
            result[f"event_logprob_{token}"] = _safe_dict_item(
                fingerprint.event_type_logits, token
            )
            result[f"event_prob_{token}"] = _safe_dict_item(
                fingerprint.event_type_probabilities, token
            )
        result["calibration_T"] = (
            float(fingerprint.calibration_T)
            if fingerprint.calibration_T is not None
            else math.nan
        )

    if signal is not None:
        result["direction"] = signal.direction
        result["confidence"] = float(signal.confidence)
        result["raw_signal_logprob_A"] = _safe_list_item(signal.raw_signal_logits, 0)
        result["raw_signal_logprob_B"] = _safe_list_item(signal.raw_signal_logits, 1)
        result["raw_signal_logprob_C"] = _safe_list_item(signal.raw_signal_logits, 2)
        result["pmi_null_logprob_A"] = _safe_list_item(signal.pmi_null_logprobs, 0)
        result["pmi_null_logprob_B"] = _safe_list_item(signal.pmi_null_logprobs, 1)
        result["pmi_null_logprob_C"] = _safe_list_item(signal.pmi_null_logprobs, 2)
        result["pmi_adjusted_logit_A"] = _safe_list_item(signal.signal_logits, 0)
        result["pmi_adjusted_logit_B"] = _safe_list_item(signal.signal_logits, 1)
        result["pmi_adjusted_logit_C"] = _safe_list_item(signal.signal_logits, 2)
        result["signal_prob_A"] = _safe_dict_item(signal.signal_probabilities, "BUY")
        result["signal_prob_B"] = _safe_dict_item(signal.signal_probabilities, "HOLD")
        result["signal_prob_C"] = _safe_dict_item(signal.signal_probabilities, "SELL")
        result["pmi_alpha_used"] = (
            float(signal.pmi_alpha_used)
            if signal.pmi_alpha_used is not None
            else math.nan
        )
        result["calibration_T"] = (
            float(signal.calibration_T)
            if signal.calibration_T is not None
            else result["calibration_T"]
        )
        result["signal_filter_forced_hold"] = signal.signal_filter_forced_hold
        result["signal_filter_reason"] = signal.signal_filter_reason
        result["signal_direction"] = signal.direction
        result["signal_confidence"] = float(signal.confidence)
        result["strategy_tag"] = signal.strategy_tag

    return result


def run_backtest(
    dataset_path: str,
    output_path: str = "output/backtest_results.csv",
    max_rows: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the full news2signal pipeline and persist detailed results.

    vLLM inference runs in batches of ``batch_size`` (default from config).
    Each batch makes these vLLM call groups:
      1. Guided fact-extraction (all articles in the batch, single call).
      2. Sentiment CoT + logits (all articles in the batch, single call).
      3. Event type direct logits (all articles in the batch, single call).
      4. Strategy logits (valid fingerprints only, single call; CoT optional).

    Parameters
    ----------
    dataset_path:
        Path to the parquet/CSV dataset.
    output_path:
        Where to write the result CSV.
    max_rows:
        Optional cap for quick tests.
    batch_size:
        Override FINGPT_BACKTEST_BATCH_SIZE from config.

    Strict mode (FINGPT_BACKTEST_STRICT_MODE):
      - fingerprint_failed       -> skip row
      - event_type_logits_failed -> skip row
      - signal_logits_failed     -> skip row
    Non-strict mode:
      - fingerprint_failed -> skip row (always - no fingerprint, no signal)
      - event_type_logits_failed -> keep row with event_type=OTHER
      - signal_logits_failed -> skip row
    """
    effective_batch_size = batch_size if batch_size is not None else FINGPT_BACKTEST_BATCH_SIZE

    df = load_dataset(dataset_path)
    rows = build_backtest_rows(df)
    if max_rows is not None:
        rows = rows[:max_rows]

    results: list[dict] = []
    total = len(rows)

    for batch_start in range(0, total, effective_batch_size):
        batch = rows[batch_start : batch_start + effective_batch_size]
        batch_end = batch_start + len(batch)
        logger.info(
            "Backtest progress: rows %d-%d / %d", batch_start + 1, batch_end, total
        )

        batch_tickers = [r["ticker"] for r in batch]

        try:
            fingerprints = extract_fingerprint_batch(
                [r["article_text"] for r in batch],
                tickers=batch_tickers,
            )
        except Exception as exc:
            logger.exception(
                "extract_fingerprint_batch failed for rows %d-%d: %s",
                batch_start + 1, batch_end, exc,
            )
            fingerprints = [None] * len(batch)

        if FINGPT_BACKTEST_STRICT_MODE:
            for j, fp in enumerate(fingerprints):
                if fp is not None and fp.event_type_method == "event_type_logits_failed":
                    fingerprints[j] = None
                    logger.info(
                        "Strict mode: skipping row %d (event_type_logits_failed).",
                        batch_start + j + 1,
                    )

        valid_pairs: list[tuple[int, object]] = [
            (j, fp) for j, fp in enumerate(fingerprints) if fp is not None
        ]
        batch_signals: list = [None] * len(batch)
        if valid_pairs:
            valid_indices, valid_fps = zip(*valid_pairs)
            try:
                raw_signals = generate_signal_batch(list(valid_fps))  # type: ignore[arg-type]
            except Exception as exc:
                logger.exception(
                    "generate_signal_batch failed for rows %d-%d: %s",
                    batch_start + 1, batch_end, exc,
                )
                raw_signals = [None] * len(valid_fps)
            for k, j in enumerate(valid_indices):
                batch_signals[j] = raw_signals[k]

        for j, row in enumerate(batch):
            fingerprint = fingerprints[j]
            signal = batch_signals[j]
            base_result = flatten_article_result(row, fingerprint=fingerprint, signal=signal)

            try:
                if fingerprint is None:
                    base_result["skipped_reason"] = "fingerprint_failed"
                    results.append(base_result)
                    continue

                if (
                    FINGPT_BACKTEST_STRICT_MODE
                    and fingerprint.event_type_method == "event_type_logits_failed"
                ):
                    base_result["skipped_reason"] = "event_type_logits_failed"
                    results.append(base_result)
                    continue

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
                    base_result["skipped_reason"] = "price_fetch_failed"
                    results.append(base_result)
                    continue

                position = _position_from_direction(signal.direction)
                strategy_return = position * realized_return

                base_result["realized_return"] = realized_return
                base_result["position"] = position
                base_result["strategy_return"] = strategy_return
                results.append(base_result)

            except Exception as exc:
                global_idx = batch_start + j + 1
                logger.exception(
                    "Failed to assemble result for row %d: %s", global_idx, exc
                )
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

    Successful rows are those where skipped_reason == "".
    Extended metrics include:
      - trade counts (long / short / neutral)
      - coverage / abstention rate
      - max drawdown
      - long/short precision
      - per-event_type grouped returns (if event_type column is present)
    """
    total_rows = int(len(results))
    successful = results[results["skipped_reason"] == ""].copy()
    successful_rows = int(len(successful))
    skipped = total_rows - successful_rows
    skip_rate = (skipped / total_rows) if total_rows else 0.0

    metrics: dict = {
        "total_rows": total_rows,
        "successful_rows": successful_rows,
        "skip_rate": skip_rate,
    }

    if successful_rows == 0:
        metrics.update({
            "direction_accuracy": 0.0,
            "long_accuracy": 0.0,
            "short_accuracy": 0.0,
            "mean_strategy_return": 0.0,
            "std_strategy_return": 0.0,
            "annualized_sharpe": 0.0,
            "total_pnl": 0.0,
            "gross_return": 0.0,
            "max_drawdown": 0.0,
            "vs_fingpt_accuracy": 0.0,
            "long_trade_count": 0,
            "short_trade_count": 0,
            "neutral_count": 0,
            "num_trades": 0,
            "coverage": 0.0,
            "abstention_rate": 0.0,
            "long_precision": 0.0,
            "short_precision": 0.0,
            "signal_filter_forced_hold_rate": 0.0,
        })
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
    short_rows = successful[successful["signal_direction"] == "short"]
    neutral_rows = successful[successful["signal_direction"] == "neutral"]

    long_count = int(len(long_rows))
    short_count = int(len(short_rows))
    neutral_count = int(len(neutral_rows))

    metrics["long_trade_count"] = long_count
    metrics["short_trade_count"] = short_count
    metrics["neutral_count"] = neutral_count
    metrics["num_trades"] = long_count + short_count
    metrics["coverage"] = (long_count + short_count) / successful_rows if successful_rows else 0.0
    metrics["abstention_rate"] = neutral_count / successful_rows if successful_rows else 0.0

    if not long_rows.empty:
        metrics["long_accuracy"] = float(
            (long_rows["realized_direction"] == "up").mean()
        )
        metrics["long_precision"] = metrics["long_accuracy"]
    else:
        metrics["long_accuracy"] = 0.0
        metrics["long_precision"] = 0.0

    if not short_rows.empty:
        metrics["short_accuracy"] = float(
            (short_rows["realized_direction"] == "down").mean()
        )
        metrics["short_precision"] = metrics["short_accuracy"]
    else:
        metrics["short_accuracy"] = 0.0
        metrics["short_precision"] = 0.0

    strategy_returns = successful["strategy_return"].astype(float)
    mean_return = float(strategy_returns.mean())
    std_return = float(strategy_returns.std(ddof=0))
    total_pnl = float(strategy_returns.sum())

    metrics["mean_strategy_return"] = mean_return
    metrics["std_strategy_return"] = std_return
    metrics["total_pnl"] = total_pnl
    metrics["gross_return"] = total_pnl

    if std_return > 0:
        metrics["annualized_sharpe"] = float((mean_return / std_return) * math.sqrt(52.0))
    else:
        metrics["annualized_sharpe"] = 0.0

    cum_returns = strategy_returns.cumsum()
    running_max = cum_returns.cummax()
    drawdowns = running_max - cum_returns
    metrics["max_drawdown"] = float(drawdowns.max()) if not drawdowns.empty else 0.0

    if "fingpt_label" in successful.columns and "signal_direction" in successful.columns:
        successful_fgt = successful.copy()
        successful_fgt["fingpt_direction"] = successful_fgt["fingpt_label"].map(
            {"up": "long", "down": "short", "neutral": "neutral"}
        )
        metrics["vs_fingpt_accuracy"] = float(
            (successful_fgt["signal_direction"] == successful_fgt["fingpt_direction"]).mean()
        )
    else:
        metrics["vs_fingpt_accuracy"] = 0.0

    if "signal_filter_forced_hold" in successful.columns:
        forced = successful["signal_filter_forced_hold"].astype(bool)
        metrics["signal_filter_forced_hold_rate"] = float(forced.mean())
    else:
        metrics["signal_filter_forced_hold_rate"] = 0.0

    if "event_type" in successful.columns:
        grouped = (
            successful.groupby("event_type")["strategy_return"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "mean_return", "count": "n_rows"})
        )
        event_breakdown: dict = {}
        for et, row in grouped.iterrows():
            event_breakdown[str(et)] = {
                "mean_return": round(float(row["mean_return"]), 6),
                "n_rows": int(row["n_rows"]),
            }
        metrics["event_type_breakdown"] = event_breakdown

    return metrics
