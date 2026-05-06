"""
Offline PMI-alpha grid search over an existing backtest CSV.

This module reuses the raw Agent 2 logits already stored in the backtest
output, so hyperparameter search does not require any model inference.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import pandas as pd

from backtest.backtester import compute_metrics
from config import (
    CALIBRATION_T,
    FINGPT_BUY_THRESHOLD,
    FINGPT_PMI_ALPHA,
    FINGPT_SELL_THRESHOLD,
    FINGPT_SIGNAL_MIN_CONFIDENCE,
    FINGPT_SIGNAL_MIN_MARGIN,
    STRATEGY_SET,
)
from vllm_logits_client import softmax

_DIRECTION_MAP = {
    "BUY": "long",
    "HOLD": "neutral",
    "SELL": "short",
}


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _extract_triplet(row: pd.Series, prefix: str) -> Optional[list[float]]:
    values: list[float] = []
    for suffix in ("A", "B", "C"):
        value = _coerce_float(row.get(f"{prefix}_{suffix}"))
        if value is None:
            return None
        values.append(value)
    return values


def _position_from_direction(direction: str) -> int:
    if direction == "long":
        return 1
    if direction == "short":
        return -1
    return 0


def _postprocess_signal(
    raw_logits: list[float],
    null_logprobs: Optional[list[float]],
    *,
    pmi_alpha: float,
    calibration_t: float,
    min_confidence: float,
    min_margin: float,
    buy_threshold: float,
    sell_threshold: float,
) -> dict[str, object]:
    adjusted_logits = list(raw_logits)
    if null_logprobs is not None and len(null_logprobs) == len(raw_logits):
        adjusted_logits = [
            raw - pmi_alpha * null_lp
            for raw, null_lp in zip(raw_logits, null_logprobs)
        ]

    probs = softmax(adjusted_logits, temperature=calibration_t)
    top_idx = probs.index(max(probs))
    top_prob = probs[top_idx]
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

    raw_best_label = ("A", "B", "C")[top_idx]
    strategy = STRATEGY_SET[top_idx]
    forced_hold = False
    filter_reason: Optional[str] = None

    if top_prob < min_confidence:
        forced_hold = True
        filter_reason = "low_confidence"
    elif margin < min_margin:
        forced_hold = True
        filter_reason = "low_margin"
    elif raw_best_label == "A" and top_prob < buy_threshold:
        forced_hold = True
        filter_reason = "buy_threshold"
    elif raw_best_label == "C" and top_prob < sell_threshold:
        forced_hold = True
        filter_reason = "sell_threshold"

    if forced_hold:
        strategy = "HOLD"

    direction = _DIRECTION_MAP[strategy]
    probabilities = dict(zip(STRATEGY_SET, probs))

    return {
        "adjusted_logits": adjusted_logits,
        "probabilities": probabilities,
        "direction": direction,
        "confidence": probabilities[strategy],
        "forced_hold": forced_hold,
        "filter_reason": filter_reason,
    }


def apply_pmi_alpha_to_results(
    results: pd.DataFrame,
    *,
    pmi_alpha: float,
    calibration_t: float = CALIBRATION_T,
    min_confidence: float = FINGPT_SIGNAL_MIN_CONFIDENCE,
    min_margin: float = FINGPT_SIGNAL_MIN_MARGIN,
    buy_threshold: float = FINGPT_BUY_THRESHOLD,
    sell_threshold: float = FINGPT_SELL_THRESHOLD,
) -> pd.DataFrame:
    """
    Recompute Agent 2 post-processing for an existing backtest DataFrame.
    """
    adjusted = results.copy()

    required_columns = [
        "pmi_adjusted_logit_A",
        "pmi_adjusted_logit_B",
        "pmi_adjusted_logit_C",
        "signal_prob_A",
        "signal_prob_B",
        "signal_prob_C",
        "direction",
        "confidence",
        "signal_direction",
        "signal_confidence",
        "signal_filter_forced_hold",
        "signal_filter_reason",
        "pmi_alpha_used",
    ]
    for column in required_columns:
        if column not in adjusted.columns:
            adjusted[column] = math.nan if "prob_" in column or "confidence" in column else ""
    if "position" not in adjusted.columns:
        adjusted["position"] = math.nan
    if "strategy_return" not in adjusted.columns:
        adjusted["strategy_return"] = math.nan
    if "skipped_reason" not in adjusted.columns:
        adjusted["skipped_reason"] = ""
    for text_column in [
        "direction",
        "signal_direction",
        "signal_filter_reason",
        "skipped_reason",
    ]:
        if text_column in adjusted.columns:
            adjusted[text_column] = adjusted[text_column].astype(object)

    for idx in adjusted.index:
        row = adjusted.loc[idx]
        raw_logits = _extract_triplet(row, "raw_signal_logprob")
        if raw_logits is None:
            continue
        null_logprobs = _extract_triplet(row, "pmi_null_logprob")

        processed = _postprocess_signal(
            raw_logits,
            null_logprobs,
            pmi_alpha=pmi_alpha,
            calibration_t=calibration_t,
            min_confidence=min_confidence,
            min_margin=min_margin,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

        adjusted.at[idx, "pmi_adjusted_logit_A"] = processed["adjusted_logits"][0]
        adjusted.at[idx, "pmi_adjusted_logit_B"] = processed["adjusted_logits"][1]
        adjusted.at[idx, "pmi_adjusted_logit_C"] = processed["adjusted_logits"][2]
        adjusted.at[idx, "signal_prob_A"] = processed["probabilities"]["BUY"]
        adjusted.at[idx, "signal_prob_B"] = processed["probabilities"]["HOLD"]
        adjusted.at[idx, "signal_prob_C"] = processed["probabilities"]["SELL"]
        adjusted.at[idx, "direction"] = processed["direction"]
        adjusted.at[idx, "confidence"] = processed["confidence"]
        adjusted.at[idx, "signal_direction"] = processed["direction"]
        adjusted.at[idx, "signal_confidence"] = processed["confidence"]
        adjusted.at[idx, "signal_filter_forced_hold"] = processed["forced_hold"]
        adjusted.at[idx, "signal_filter_reason"] = processed["filter_reason"] or ""
        adjusted.at[idx, "pmi_alpha_used"] = pmi_alpha

        realized_return = _coerce_float(row.get("realized_return"))
        if realized_return is None:
            continue

        position = _position_from_direction(str(processed["direction"]))
        adjusted.at[idx, "position"] = position
        adjusted.at[idx, "strategy_return"] = position * realized_return
        skipped_reason = str(row.get("skipped_reason", "") or "")
        if skipped_reason == "price_fetch_failed":
            continue
        if skipped_reason in {"", "signal_failed"}:
            adjusted.at[idx, "skipped_reason"] = ""

    return adjusted


def run_pmi_alpha_grid_search(
    results_or_path: pd.DataFrame | str,
    *,
    alphas: list[float],
    output_path: Optional[str] = None,
    detailed_output_dir: Optional[str] = None,
    calibration_t: float = CALIBRATION_T,
    min_confidence: float = FINGPT_SIGNAL_MIN_CONFIDENCE,
    min_margin: float = FINGPT_SIGNAL_MIN_MARGIN,
    buy_threshold: float = FINGPT_BUY_THRESHOLD,
    sell_threshold: float = FINGPT_SELL_THRESHOLD,
) -> pd.DataFrame:
    """
    Evaluate multiple PMI alpha values on an existing backtest CSV/DataFrame.
    """
    if isinstance(results_or_path, pd.DataFrame):
        base_results = results_or_path.copy()
    else:
        base_results = pd.read_csv(results_or_path)

    summary_rows: list[dict[str, object]] = []
    original_direction = (
        base_results["signal_direction"].astype(str)
        if "signal_direction" in base_results.columns
        else pd.Series([""] * len(base_results), index=base_results.index)
    )
    for alpha in alphas:
        adjusted = apply_pmi_alpha_to_results(
            base_results,
            pmi_alpha=alpha,
            calibration_t=calibration_t,
            min_confidence=min_confidence,
            min_margin=min_margin,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        metrics = compute_metrics(adjusted)
        raw_signal_mask = adjusted[[
            "raw_signal_logprob_A",
            "raw_signal_logprob_B",
            "raw_signal_logprob_C",
        ]].notna().all(axis=1)
        signal_rows = adjusted.loc[raw_signal_mask].copy()
        signal_direction = signal_rows["signal_direction"].astype(str)
        summary_row: dict[str, object] = {
            "pmi_alpha": alpha,
            "calibration_T": calibration_t,
            "signal_min_confidence": min_confidence,
            "signal_min_margin": min_margin,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "raw_rows": int(len(base_results)),
            "signal_rows": int(len(signal_rows)),
            "signal_long_count": int((signal_direction == "long").sum()),
            "signal_short_count": int((signal_direction == "short").sum()),
            "signal_neutral_count": int((signal_direction == "neutral").sum()),
            "signal_changed_count": int(
                (
                    signal_rows["signal_direction"].astype(str)
                    != original_direction.loc[signal_rows.index]
                ).sum()
            ),
        }
        summary_row["signal_changed_rate"] = (
            summary_row["signal_changed_count"] / summary_row["signal_rows"]
            if summary_row["signal_rows"]
            else 0.0
        )
        summary_row.update(metrics)
        summary_rows.append(summary_row)

        if detailed_output_dir:
            os.makedirs(detailed_output_dir, exist_ok=True)
            alpha_slug = str(alpha).replace("-", "neg_").replace(".", "_")
            adjusted.to_csv(
                os.path.join(detailed_output_dir, f"backtest_alpha_{alpha_slug}.csv"),
                index=False,
            )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["rank_total_pnl"] = summary["total_pnl"].rank(
            method="dense", ascending=False
        )
        summary["rank_sharpe"] = summary["annualized_sharpe"].rank(
            method="dense", ascending=False
        )
        summary["rank_direction_accuracy"] = summary["direction_accuracy"].rank(
            method="dense", ascending=False
        )

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(output_path, index=False)

    return summary


def parse_alpha_grid(alpha_spec: str) -> list[float]:
    """
    Parse a comma-separated alpha list like ``"0,0.25,0.5,1.0"``.
    """
    alphas: list[float] = []
    for chunk in alpha_spec.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        alphas.append(float(stripped))
    if not alphas:
        return [FINGPT_PMI_ALPHA]
    return alphas
