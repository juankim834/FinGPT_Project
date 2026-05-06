"""
agent2/schema.py — Pydantic schema for Agent 2 output.

TradingSignal is backward compatible with the previous version.
Optional logit/probability fields support logits-based confidence calculation.

Changes from previous version:
- cot is now optional (default "") to support no-CoT mode.
- strategy_tag is always set to "event_driven" by the pipeline (not predicted
  by the model).  The Literal type still validates allowed values.
- signal_filter_forced_hold and signal_filter_reason record whether a
  confidence/margin filter overrode the model's top choice.
- raw_signal_logits stores pre-PMI logits for diagnostics.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TradingSignal(BaseModel):
    ticker: str
    direction: Literal["long", "short", "neutral"]
    strategy_tag: Literal["momentum", "mean_reversion", "event_driven", "macro", "none"]
    confidence: float = Field(ge=0.0, le=1.0)

    # Chain-of-thought reasoning text; empty string in no-CoT mode.
    cot: str = ""

    # -----------------------------------------------------------------------
    # Logits-derived fields (optional for backward compatibility)
    # -----------------------------------------------------------------------

    # PMI-adjusted logits (after subtracting alpha * null_logprobs).
    signal_logits: Optional[list[float]] = None

    # Raw logits before PMI correction — stored for diagnostics.
    raw_signal_logits: Optional[list[float]] = None

    pmi_null_logprobs: Optional[list[float]] = None
    pmi_alpha_used: Optional[float] = None

    # Calibrated probability vector, e.g. {"BUY": 0.72, "HOLD": 0.18, "SELL": 0.10}.
    signal_probabilities: Optional[dict[str, float]] = None

    # Temperature used for softmax calibration.
    calibration_T: Optional[float] = None

    # -----------------------------------------------------------------------
    # Signal filter fields — record whether a threshold override was applied
    # -----------------------------------------------------------------------

    # True when a confidence/margin/direction threshold forced the direction
    # to HOLD regardless of the model's top choice.
    signal_filter_forced_hold: bool = False

    # Human-readable reason for the forced HOLD, e.g. "low_confidence",
    # "low_margin", "buy_threshold", "sell_threshold".  None when no filter.
    signal_filter_reason: Optional[str] = None
