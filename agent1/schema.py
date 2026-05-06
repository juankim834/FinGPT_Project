"""
agent1/schema.py — Pydantic schema for Agent 1 output.

Sentiment is expressed as a 3-class label (POSITIVE / NEGATIVE / NEUTRAL)
derived deterministically from LLM-reported logits via softmax post-processing.

Event type is expressed as one of 7 concrete categories (EARNINGS, GUIDANCE,
ANALYST_RATING, LEGAL_REGULATORY, MNA, PRODUCT_BUSINESS, MACRO) or OTHER
when confidence/margin thresholds are not met.  It is derived from A-G token
logprobs — OTHER is never a candidate score token (it is assigned by Python).

The NewsFingerprint carries the full probability vectors and calibration
temperature so downstream consumers can reproduce probability calculations.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# 3-class sentiment aligned with the logits prompt order:
#   index 0 → POSITIVE,  index 1 → NEGATIVE,  index 2 → NEUTRAL
SentimentLabel = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]

# Numeric score derived from sentiment label (for backward compat with backtester).
_SENTIMENT_SCORE: dict[str, float] = {
    "POSITIVE": 1.0,
    "NEGATIVE": -1.0,
    "NEUTRAL": 0.0,
}


class NewsFingerprint(BaseModel):
    # -----------------------------------------------------------------------
    # Fact-extraction fields
    # -----------------------------------------------------------------------
    source: str
    published_at: str
    headline: str
    companies_named: list[str]

    # event_keywords kept for backward compatibility; defaults to empty list.
    # In the new pipeline this is set to [event_type] by _assemble_fingerprint.
    event_keywords: list[str] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # Sentiment fields — populated from logits post-processing
    # -----------------------------------------------------------------------
    sentiment_label: SentimentLabel

    # Scalar score kept for backward compatibility with backtester metrics.
    # Derived from label: POSITIVE=1.0, NEUTRAL=0.0, NEGATIVE=-1.0.
    sentiment_score: float = Field(ge=-1.0, le=1.0)

    # max(softmax(logits / calibration_T)) — true calibrated confidence.
    sentiment_confidence: float = Field(ge=0.0, le=1.0)

    # Full calibrated probability vector, e.g.
    #   {"POSITIVE": 0.87, "NEGATIVE": 0.08, "NEUTRAL": 0.05}
    sentiment_probabilities: dict[str, float] = Field(default_factory=dict)

    # Temperature used for softmax calibration (from config.CALIBRATION_T).
    calibration_T: float = Field(default=1.0, ge=0.0)

    # Raw logits reported by the LLM (before softmax), stored for auditing.
    sentiment_logits: Optional[list[float]] = None

    # -----------------------------------------------------------------------
    # Event type fields — populated from A-G logits post-processing
    # -----------------------------------------------------------------------

    # Concrete event category or "OTHER" when thresholds are not met.
    event_type: str = "OTHER"

    # Softmax probability of the top event_type class (None if logits failed).
    event_type_confidence: Optional[float] = None

    # top_prob − second_prob margin (None if logits failed).
    event_type_margin: Optional[float] = None

    # How event_type was assigned:
    #   "logits_accepted"           — concrete label accepted
    #   "abstained_low_confidence"  — top_prob < min_confidence → OTHER
    #   "abstained_low_margin"      — margin < min_margin → OTHER
    #   "event_type_logits_failed"  — vLLM call failed → OTHER
    event_type_method: Optional[str] = None

    # Raw log-probabilities for each A-G token, keyed by token letter.
    event_type_logits: Optional[dict[str, float]] = None

    # Softmax probability for each A-G token after calibration.
    event_type_probabilities: Optional[dict[str, float]] = None

    # Second-best concrete event category (None if logits failed).
    secondary_event_type: Optional[str] = None

    # Softmax probability of the second-best event_type class.
    secondary_event_type_confidence: Optional[float] = None

    # -----------------------------------------------------------------------
    # Original article text
    # -----------------------------------------------------------------------
    # Stored so Agent 2 can access context without a separate argument.
    article_text: str = ""

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("event_keywords")
    @classmethod
    def keywords_must_be_lowercase(cls, value: list[str]) -> list[str]:
        return [keyword.lower() for keyword in value]
