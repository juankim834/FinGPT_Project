"""
tests/test_extractor.py — Unit tests for Agent 1 deterministic logic.

These tests cover the pure-Python post-processing functions that do NOT
require a GPU or vLLM:
  - _rank_probabilities
  - _apply_classification_thresholds
  - _process_event_type_result
  - EVENT_TYPE_CLASSES (no OTHER token)
  - EVENT_TYPE_MAP (A-G → concrete labels)
  - _assemble_fingerprint (ticker fallback)

vLLM is not imported; get_real_choice_logits is mocked where needed so that
_process_event_type_result can be tested end-to-end.
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from config import EVENT_TYPE_CLASSES, EVENT_TYPE_MAP


# ---------------------------------------------------------------------------
# 1. EVENT_TYPE_CLASSES — no OTHER token
# ---------------------------------------------------------------------------

class TestEventTypeClasses:
    def test_no_other_in_classes(self):
        """OTHER must never appear as a candidate score token."""
        assert "OTHER" not in EVENT_TYPE_CLASSES

    def test_no_unclear_in_classes(self):
        assert "UNCLEAR" not in EVENT_TYPE_CLASSES

    def test_seven_concrete_tokens(self):
        assert len(EVENT_TYPE_CLASSES) == 7
        assert set(EVENT_TYPE_CLASSES) == {"A", "B", "C", "D", "E", "F", "G"}


# ---------------------------------------------------------------------------
# 2. EVENT_TYPE_MAP — A-G → concrete labels
# ---------------------------------------------------------------------------

class TestEventTypeMap:
    def test_all_tokens_mapped(self):
        for token in EVENT_TYPE_CLASSES:
            assert token in EVENT_TYPE_MAP

    def test_correct_labels(self):
        assert EVENT_TYPE_MAP["A"] == "EARNINGS"
        assert EVENT_TYPE_MAP["B"] == "GUIDANCE"
        assert EVENT_TYPE_MAP["C"] == "ANALYST_RATING"
        assert EVENT_TYPE_MAP["D"] == "LEGAL_REGULATORY"
        assert EVENT_TYPE_MAP["E"] == "MNA"
        assert EVENT_TYPE_MAP["F"] == "PRODUCT_BUSINESS"
        assert EVENT_TYPE_MAP["G"] == "MACRO"

    def test_no_other_in_values(self):
        assert "OTHER" not in EVENT_TYPE_MAP.values()


# ---------------------------------------------------------------------------
# 3. _rank_probabilities
# ---------------------------------------------------------------------------

class TestRankProbabilities:
    @pytest.fixture(autouse=True)
    def _import(self):
        from agent1.extractor import _rank_probabilities
        self.fn = _rank_probabilities

    def test_top_idx_is_argmax(self):
        probs = [0.1, 0.6, 0.3]
        labels = ["A", "B", "C"]
        top_idx, second_idx, top_prob, second_prob, margin = self.fn(probs, labels)
        assert top_idx == 1
        assert abs(top_prob - 0.6) < 1e-9

    def test_second_idx_is_second_argmax(self):
        probs = [0.1, 0.6, 0.3]
        labels = ["A", "B", "C"]
        top_idx, second_idx, top_prob, second_prob, margin = self.fn(probs, labels)
        assert second_idx == 2
        assert abs(second_prob - 0.3) < 1e-9

    def test_margin_equals_top_minus_second(self):
        probs = [0.1, 0.6, 0.3]
        labels = ["A", "B", "C"]
        top_idx, second_idx, top_prob, second_prob, margin = self.fn(probs, labels)
        assert abs(margin - (top_prob - second_prob)) < 1e-9

    def test_ties_handled_gracefully(self):
        probs = [0.5, 0.5]
        labels = ["A", "B"]
        top_idx, second_idx, top_prob, second_prob, margin = self.fn(probs, labels)
        assert top_prob == 0.5
        assert second_prob == 0.5
        assert abs(margin) < 1e-9


# ---------------------------------------------------------------------------
# 4. _apply_classification_thresholds
# ---------------------------------------------------------------------------

class TestApplyClassificationThresholds:
    @pytest.fixture(autouse=True)
    def _import(self):
        from agent1.extractor import _apply_classification_thresholds
        self.fn = _apply_classification_thresholds

    def test_accepted_when_above_both_thresholds(self):
        label, method = self.fn("A", 0.8, 0.3, 0.5, 0.1, EVENT_TYPE_MAP)
        assert label == "EARNINGS"
        assert method == "logits_accepted"

    def test_other_when_below_confidence_threshold(self):
        label, method = self.fn("A", 0.3, 0.2, 0.5, 0.1, EVENT_TYPE_MAP)
        assert label == "OTHER"
        assert method == "abstained_low_confidence"

    def test_other_when_below_margin_threshold(self):
        # top_prob (0.6) >= min_confidence (0.5) but margin (0.05) < min_margin (0.1)
        label, method = self.fn("B", 0.6, 0.05, 0.5, 0.1, EVENT_TYPE_MAP)
        assert label == "OTHER"
        assert method == "abstained_low_margin"

    def test_confidence_check_takes_priority_over_margin(self):
        # Both fail: confidence check should fire first.
        label, method = self.fn("C", 0.2, 0.01, 0.5, 0.1, EVENT_TYPE_MAP)
        assert label == "OTHER"
        assert method == "abstained_low_confidence"

    def test_zero_thresholds_always_accept(self):
        """Default thresholds (0.0) should never trigger abstention."""
        for token in EVENT_TYPE_CLASSES:
            label, method = self.fn(token, 0.001, 0.0, 0.0, 0.0, EVENT_TYPE_MAP)
            assert label == EVENT_TYPE_MAP[token]
            assert method == "logits_accepted"

    def test_all_tokens_map_correctly(self):
        for token, expected_label in EVENT_TYPE_MAP.items():
            label, method = self.fn(token, 1.0, 1.0, 0.0, 0.0, EVENT_TYPE_MAP)
            assert label == expected_label
            assert method == "logits_accepted"


# ---------------------------------------------------------------------------
# 5. _process_event_type_result — secondary event type
# ---------------------------------------------------------------------------

class TestProcessEventTypeResult:
    @pytest.fixture(autouse=True)
    def _import(self):
        from agent1.extractor import _process_event_type_result
        self.fn = _process_event_type_result

    def _make_result(self, logits, parse_success=True):
        return {
            "logits": logits,
            "choices": EVENT_TYPE_CLASSES,
            "raw_output": "",
            "thinking": "",
            "parse_success": parse_success,
        }

    def test_top_label_accepted_with_zero_thresholds(self):
        # Uniform logits: softmax gives equal probs, top token is first in sort order.
        # Use skewed logits so A wins clearly.
        logits = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = self.fn(
            self._make_result(logits),
            article_text="test",
            save_debug=False,
            min_confidence=0.0,
            min_margin=0.0,
        )
        assert result["event_type"] == "EARNINGS"
        assert result["event_type_method"] == "logits_accepted"
        assert result["event_type_confidence"] is not None
        assert result["event_type_confidence"] > 0.5

    def test_secondary_event_type_is_second_best(self):
        # A wins, G is second.
        logits = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0]
        result = self.fn(
            self._make_result(logits),
            article_text="test",
            save_debug=False,
            min_confidence=0.0,
            min_margin=0.0,
        )
        assert result["event_type"] == "EARNINGS"
        assert result["secondary_event_type"] == "MACRO"
        assert result["secondary_event_type_confidence"] is not None

    def test_low_confidence_assigns_other(self):
        # Near-uniform: top prob will be ~1/7 ≈ 0.143; set threshold above that.
        logits = [1.0] * 7
        result = self.fn(
            self._make_result(logits),
            article_text="test",
            save_debug=False,
            min_confidence=0.5,
            min_margin=0.0,
        )
        assert result["event_type"] == "OTHER"
        assert result["event_type_method"] == "abstained_low_confidence"

    def test_low_margin_assigns_other(self):
        # Two tokens tie closely; margin will be near 0.
        logits = [5.0, 4.99, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = self.fn(
            self._make_result(logits),
            article_text="test",
            save_debug=False,
            min_confidence=0.0,
            min_margin=0.4,   # margin will be tiny
        )
        assert result["event_type"] == "OTHER"
        assert result["event_type_method"] == "abstained_low_margin"

    def test_parse_failure_assigns_other_with_failed_method(self):
        failed_result = self._make_result(None, parse_success=False)
        result = self.fn(
            failed_result,
            article_text="test",
            save_debug=False,
        )
        assert result["event_type"] == "OTHER"
        assert result["event_type_method"] == "event_type_logits_failed"
        assert result["event_type_logits"] is None
        assert result["event_type_probabilities"] is None
        assert result["secondary_event_type"] is None

    def test_logits_dict_keyed_by_token(self):
        logits = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = self.fn(
            self._make_result(logits),
            article_text="test",
            save_debug=False,
            min_confidence=0.0,
            min_margin=0.0,
        )
        assert set(result["event_type_logits"].keys()) == set(EVENT_TYPE_CLASSES)
        assert set(result["event_type_probabilities"].keys()) == set(EVENT_TYPE_CLASSES)


# ---------------------------------------------------------------------------
# 6. _assemble_fingerprint — ticker fallback
# ---------------------------------------------------------------------------

class TestAssembleFingerprint:
    @pytest.fixture(autouse=True)
    def _import(self):
        from agent1.extractor import _assemble_fingerprint
        self.fn = _assemble_fingerprint

    def _make_inputs(self, companies_named=None):
        extracted = {
            "source": "Reuters",
            "published_at": "2026-05-01T12:00:00Z",
            "headline": "Apple Q2 earnings beat expectations",
            "companies_named": companies_named if companies_named is not None else ["AAPL"],
            "event_keywords": [],
        }
        sentiment = {
            "sentiment_label": "POSITIVE",
            "sentiment_score": 1.0,
            "sentiment_confidence": 0.85,
            "sentiment_probabilities": {"POSITIVE": 0.85, "NEGATIVE": 0.08, "NEUTRAL": 0.07},
            "sentiment_logits": [-5.0, -2.0, -1.0],
            "calibration_T": 1.2,
        }
        event_type = {
            "event_type": "EARNINGS",
            "event_type_confidence": 0.75,
            "event_type_margin": 0.30,
            "event_type_method": "logits_accepted",
            "event_type_logits": {"A": 4.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0, "G": 1.0},
            "event_type_probabilities": {"A": 0.75, "B": 0.04, "C": 0.04, "D": 0.04, "E": 0.04, "F": 0.04, "G": 0.04},
            "secondary_event_type": "GUIDANCE",
            "secondary_event_type_confidence": 0.04,
        }
        return extracted, sentiment, event_type

    def test_normal_assembly_works(self):
        extracted, sentiment, event_type = self._make_inputs()
        fp = self.fn(extracted, sentiment, event_type, "article text")
        assert fp.headline == "Apple Q2 earnings beat expectations"
        assert fp.event_type == "EARNINGS"
        assert fp.sentiment_label == "POSITIVE"
        assert "AAPL" in fp.companies_named

    def test_ticker_fallback_fills_empty_companies_named(self):
        extracted, sentiment, event_type = self._make_inputs(companies_named=[])
        fp = self.fn(extracted, sentiment, event_type, "article text", fallback_ticker="AAPL")
        assert fp.companies_named == ["AAPL"]

    def test_no_fallback_with_empty_companies_gives_empty(self):
        """Without ticker fallback, empty companies_named passes (validator removed)."""
        extracted, sentiment, event_type = self._make_inputs(companies_named=[])
        fp = self.fn(extracted, sentiment, event_type, "article text", fallback_ticker=None)
        assert fp.companies_named == []

    def test_event_keywords_set_to_event_type(self):
        extracted, sentiment, event_type = self._make_inputs()
        fp = self.fn(extracted, sentiment, event_type, "article text")
        assert fp.event_keywords == ["earnings"]  # lowercased event_type

    def test_existing_companies_not_overridden_by_fallback(self):
        extracted, sentiment, event_type = self._make_inputs(companies_named=["MSFT"])
        fp = self.fn(extracted, sentiment, event_type, "article text", fallback_ticker="AAPL")
        assert fp.companies_named == ["MSFT"]
