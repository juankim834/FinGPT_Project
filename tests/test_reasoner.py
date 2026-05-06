"""
tests/test_reasoner.py — Unit tests for Agent 2 deterministic logic.

These tests cover the pure-Python post-processing in agent2.reasoner that
does NOT require a GPU or vLLM:
  - PMI alpha correction (alpha=0 → no change, alpha=1 → raw - prior)
  - temperature softmax
  - confidence threshold forcing HOLD
  - margin threshold forcing HOLD
  - asymmetric buy/sell threshold
  - strategy_tag always equals "event_driven"
  - signal_filter_forced_hold=True recorded when filter applies
  - A/B/C → long/neutral/short direction mapping

vLLM is not imported.  _process_signal_result is tested by patching
module-level globals (_null_logprobs, FINGPT_SIGNAL_MIN_CONFIDENCE, etc.)
via monkeypatching or direct function calls with controlled inputs.
"""

import math
from typing import Optional
from unittest.mock import patch

import pytest

from agent1.schema import NewsFingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fingerprint(
    companies_named=None,
    event_type="EARNINGS",
    event_type_confidence=0.75,
    event_type_margin=0.30,
    event_type_method="logits_accepted",
) -> NewsFingerprint:
    return NewsFingerprint(
        ticker="AAPL",
        source="test",
        published_at="2026-05-01T12:00:00Z",
        headline="Test headline for trading signal.",
        companies_named=companies_named or ["AAPL"],
        sentiment_label="POSITIVE",
        sentiment_score=1.0,
        sentiment_confidence=0.85,
        sentiment_probabilities={"POSITIVE": 0.85, "NEGATIVE": 0.08, "NEUTRAL": 0.07},
        event_type=event_type,
        event_type_confidence=event_type_confidence,
        event_type_margin=event_type_margin,
        event_type_method=event_type_method,
        article_text="Test article text.",
    )


def _make_logits_result(logits, parse_success=True, thinking=""):
    return {
        "logits": logits,
        "choices": ["A", "B", "C"],
        "raw_output": thinking,
        "thinking": thinking,
        "parse_success": parse_success,
    }


def _call_process_signal(
    fingerprint,
    logits,
    null_logprobs=None,
    pmi_alpha=1.0,
    min_confidence=0.0,
    min_margin=0.0,
    buy_threshold=0.0,
    sell_threshold=0.0,
    calibration_t=1.0,
):
    """
    Call _process_signal_result with controlled config values injected
    via monkeypatching of module-level constants.
    """
    import agent2.reasoner as reasoner_mod

    result = _make_logits_result(logits)

    with (
        patch.object(reasoner_mod, "_null_logprobs", null_logprobs),
        patch("agent2.reasoner.FINGPT_PMI_ALPHA", pmi_alpha),
        patch("agent2.reasoner.FINGPT_SIGNAL_MIN_CONFIDENCE", min_confidence),
        patch("agent2.reasoner.FINGPT_SIGNAL_MIN_MARGIN", min_margin),
        patch("agent2.reasoner.FINGPT_BUY_THRESHOLD", buy_threshold),
        patch("agent2.reasoner.FINGPT_SELL_THRESHOLD", sell_threshold),
        patch("agent2.reasoner.CALIBRATION_T", calibration_t),
        patch("agent2.reasoner._log_thinking"),
        patch("agent2.reasoner._save_failure_diagnostic"),
    ):
        return reasoner_mod._process_signal_result(fingerprint, result)


# ---------------------------------------------------------------------------
# 1. A/B/C → long/neutral/short direction mapping
# ---------------------------------------------------------------------------

class TestDirectionMapping:
    def test_a_maps_to_long(self):
        fp = _make_fingerprint()
        signal = _call_process_signal(fp, logits=[5.0, 1.0, 1.0])
        assert signal is not None
        assert signal.direction == "long"

    def test_b_maps_to_neutral(self):
        fp = _make_fingerprint()
        signal = _call_process_signal(fp, logits=[1.0, 5.0, 1.0])
        assert signal is not None
        assert signal.direction == "neutral"

    def test_c_maps_to_short(self):
        fp = _make_fingerprint()
        signal = _call_process_signal(fp, logits=[1.0, 1.0, 5.0])
        assert signal is not None
        assert signal.direction == "short"


# ---------------------------------------------------------------------------
# 2. strategy_tag is always "event_driven"
# ---------------------------------------------------------------------------

class TestStrategyTag:
    def test_buy_signal_has_event_driven_tag(self):
        fp = _make_fingerprint()
        signal = _call_process_signal(fp, logits=[5.0, 1.0, 1.0])
        assert signal is not None
        assert signal.strategy_tag == "event_driven"

    def test_sell_signal_has_event_driven_tag(self):
        fp = _make_fingerprint()
        signal = _call_process_signal(fp, logits=[1.0, 1.0, 5.0])
        assert signal is not None
        assert signal.strategy_tag == "event_driven"

    def test_hold_signal_has_event_driven_tag(self):
        fp = _make_fingerprint()
        signal = _call_process_signal(fp, logits=[1.0, 5.0, 1.0])
        assert signal is not None
        assert signal.strategy_tag == "event_driven"


# ---------------------------------------------------------------------------
# 3. PMI alpha correction
# ---------------------------------------------------------------------------

class TestPmiAlpha:
    def test_alpha_zero_leaves_logits_unchanged(self):
        """alpha=0 → adjusted = raw; null prior has no effect."""
        fp = _make_fingerprint()
        raw = [3.0, 1.0, 1.0]
        null = [1.0, 1.0, 1.0]
        signal = _call_process_signal(fp, logits=raw, null_logprobs=null, pmi_alpha=0.0)
        assert signal is not None
        # With alpha=0 adjusted=raw → A is top → long
        assert signal.direction == "long"
        # signal_logits should equal raw (no correction)
        assert signal.signal_logits == raw
        assert signal.raw_signal_logits == raw

    def test_alpha_one_subtracts_null(self):
        """alpha=1 → adjusted[i] = raw[i] - null[i]."""
        fp = _make_fingerprint()
        raw = [3.0, 2.0, 1.0]
        null = [2.0, 0.0, 0.0]
        # adjusted = [1.0, 2.0, 1.0] → B wins → neutral
        signal = _call_process_signal(fp, logits=raw, null_logprobs=null, pmi_alpha=1.0)
        assert signal is not None
        assert signal.direction == "neutral"
        assert signal.raw_signal_logits == raw

    def test_alpha_partial_scales_correction(self):
        """alpha=0.5 → adjusted[i] = raw[i] - 0.5 * null[i]."""
        fp = _make_fingerprint()
        raw = [3.0, 1.0, 1.0]
        null = [4.0, 0.0, 0.0]
        # adjusted = [3.0 - 2.0, 1.0, 1.0] = [1.0, 1.0, 1.0] → tied → A wins (first)
        signal = _call_process_signal(fp, logits=raw, null_logprobs=null, pmi_alpha=0.5)
        assert signal is not None
        # [1.0, 1.0, 1.0] → softmax = uniform → argmax = index 0 → A → long
        assert signal.direction == "long"

    def test_no_null_logprobs_uses_raw(self):
        """When _null_logprobs is None, no correction is applied."""
        fp = _make_fingerprint()
        raw = [5.0, 1.0, 1.0]
        signal = _call_process_signal(fp, logits=raw, null_logprobs=None, pmi_alpha=1.0)
        assert signal is not None
        assert signal.direction == "long"
        assert signal.signal_logits == raw


# ---------------------------------------------------------------------------
# 4. Temperature softmax
# ---------------------------------------------------------------------------

class TestTemperatureSoftmax:
    def test_high_temperature_softens_distribution(self):
        """Higher T → flatter probs → lower confidence."""
        fp = _make_fingerprint()
        logits = [5.0, 1.0, 1.0]

        signal_sharp = _call_process_signal(fp, logits=logits, calibration_t=0.5)
        signal_soft = _call_process_signal(fp, logits=logits, calibration_t=2.0)

        assert signal_sharp is not None and signal_soft is not None
        # Both should pick A (long) but sharp has higher confidence.
        assert signal_sharp.direction == "long"
        assert signal_soft.direction == "long"
        assert signal_sharp.confidence > signal_soft.confidence

    def test_unit_temperature_is_standard_softmax(self):
        """T=1.0 should match manual softmax."""
        from vllm_logits_client import softmax as vllm_softmax

        fp = _make_fingerprint()
        logits = [2.0, 1.0, 0.5]
        signal = _call_process_signal(fp, logits=logits, calibration_t=1.0)
        expected_probs = vllm_softmax(logits, temperature=1.0)
        assert signal is not None
        # Signal probabilities should match (within float rounding).
        for i, key in enumerate(["BUY", "HOLD", "SELL"]):
            assert abs(signal.signal_probabilities[key] - round(expected_probs[i], 6)) < 1e-5


# ---------------------------------------------------------------------------
# 5. Confidence threshold forces HOLD
# ---------------------------------------------------------------------------

class TestConfidenceThreshold:
    def test_confidence_below_threshold_forces_hold(self):
        """top_prob < min_confidence → direction forced to neutral."""
        fp = _make_fingerprint()
        # Near-uniform: top prob ≈ 1/3 ≈ 0.33
        logits = [1.0, 0.9, 0.8]
        signal = _call_process_signal(
            fp, logits=logits, min_confidence=0.5, min_margin=0.0
        )
        assert signal is not None
        assert signal.direction == "neutral"
        assert signal.signal_filter_forced_hold is True
        assert signal.signal_filter_reason == "low_confidence"

    def test_confidence_at_threshold_does_not_force_hold(self):
        """top_prob == min_confidence should NOT force hold (strict < check)."""
        from vllm_logits_client import softmax as vllm_softmax

        logits = [5.0, 1.0, 1.0]
        probs = vllm_softmax(logits, temperature=1.0)
        top_prob = max(probs)  # top prob for these logits

        fp = _make_fingerprint()
        signal = _call_process_signal(
            fp, logits=logits, min_confidence=top_prob, min_margin=0.0
        )
        # top_prob is NOT < top_prob (equal), so no forced hold
        assert signal is not None
        assert signal.signal_filter_forced_hold is False


# ---------------------------------------------------------------------------
# 6. Margin threshold forces HOLD
# ---------------------------------------------------------------------------

class TestMarginThreshold:
    def test_margin_below_threshold_forces_hold(self):
        """margin < min_margin → direction forced to neutral."""
        fp = _make_fingerprint()
        # Close contest between A and B: small margin
        logits = [2.0, 1.99, 0.0]
        signal = _call_process_signal(
            fp, logits=logits, min_confidence=0.0, min_margin=0.4
        )
        assert signal is not None
        assert signal.direction == "neutral"
        assert signal.signal_filter_forced_hold is True
        assert signal.signal_filter_reason == "low_margin"

    def test_confidence_checked_before_margin(self):
        """If both fail, reason should be low_confidence (checked first)."""
        logits = [1.001, 1.000, 0.5]  # both low confidence and small margin
        fp = _make_fingerprint()
        signal = _call_process_signal(
            fp, logits=logits, min_confidence=0.9, min_margin=0.4
        )
        assert signal is not None
        assert signal.signal_filter_reason == "low_confidence"


# ---------------------------------------------------------------------------
# 7. Asymmetric buy/sell threshold
# ---------------------------------------------------------------------------

class TestDirectionThresholds:
    def test_buy_threshold_forces_hold_when_a_too_weak(self):
        """prob[A] < buy_threshold → HOLD even when A is top."""
        from vllm_logits_client import softmax as vllm_softmax

        logits = [2.0, 1.5, 1.0]
        probs = vllm_softmax(logits, temperature=1.0)
        prob_a = probs[0]

        fp = _make_fingerprint()
        # Set buy_threshold just above actual prob_a.
        signal = _call_process_signal(
            fp, logits=logits, buy_threshold=prob_a + 0.01, sell_threshold=0.0
        )
        assert signal is not None
        assert signal.direction == "neutral"
        assert signal.signal_filter_forced_hold is True
        assert signal.signal_filter_reason == "buy_threshold"

    def test_sell_threshold_forces_hold_when_c_too_weak(self):
        """prob[C] < sell_threshold → HOLD even when C is top."""
        from vllm_logits_client import softmax as vllm_softmax

        logits = [1.0, 1.5, 2.0]
        probs = vllm_softmax(logits, temperature=1.0)
        prob_c = probs[2]

        fp = _make_fingerprint()
        signal = _call_process_signal(
            fp, logits=logits, buy_threshold=0.0, sell_threshold=prob_c + 0.01
        )
        assert signal is not None
        assert signal.direction == "neutral"
        assert signal.signal_filter_forced_hold is True
        assert signal.signal_filter_reason == "sell_threshold"

    def test_sell_threshold_not_applied_to_buy_signal(self):
        """sell_threshold should not force HOLD when top is A."""
        logits = [5.0, 1.0, 0.1]
        fp = _make_fingerprint()
        signal = _call_process_signal(
            fp, logits=logits, buy_threshold=0.0, sell_threshold=0.99
        )
        assert signal is not None
        # A wins; sell_threshold only applies when C is top
        assert signal.direction == "long"

    def test_no_forced_hold_when_thresholds_zero(self):
        """Default thresholds (0.0) should never force hold."""
        for logits in [[5.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 5.0]]:
            fp = _make_fingerprint()
            signal = _call_process_signal(fp, logits=logits)
            assert signal is not None
            assert signal.signal_filter_forced_hold is False


# ---------------------------------------------------------------------------
# 8. Parse failure returns None
# ---------------------------------------------------------------------------

class TestParseFailure:
    def test_parse_failure_returns_none(self):
        import agent2.reasoner as reasoner_mod

        fp = _make_fingerprint()
        failed_result = {
            "logits": None,
            "choices": ["A", "B", "C"],
            "raw_output": "",
            "thinking": "",
            "parse_success": False,
        }
        with (
            patch("agent2.reasoner._log_thinking"),
            patch("agent2.reasoner._save_failure_diagnostic"),
        ):
            signal = reasoner_mod._process_signal_result(fp, failed_result)
        assert signal is None


# ---------------------------------------------------------------------------
# 9. Raw and adjusted logits are stored separately
# ---------------------------------------------------------------------------

class TestLogitsStorage:
    def test_raw_and_adjusted_logits_differ_with_pmi(self):
        fp = _make_fingerprint()
        raw = [3.0, 1.0, 1.0]
        null = [1.0, 0.0, 0.0]
        signal = _call_process_signal(fp, logits=raw, null_logprobs=null, pmi_alpha=1.0)
        assert signal is not None
        assert signal.raw_signal_logits == raw
        assert signal.signal_logits == [2.0, 1.0, 1.0]  # raw - 1.0 * null
        assert signal.pmi_null_logprobs == null
        assert signal.pmi_alpha_used == 1.0

    def test_raw_equals_adjusted_when_alpha_zero(self):
        fp = _make_fingerprint()
        raw = [3.0, 1.0, 1.0]
        null = [2.0, 2.0, 2.0]
        signal = _call_process_signal(fp, logits=raw, null_logprobs=null, pmi_alpha=0.0)
        assert signal is not None
        assert signal.raw_signal_logits == raw
        assert signal.signal_logits == raw
        assert signal.pmi_null_logprobs == null
        assert signal.pmi_alpha_used == 0.0
