import math

from agent1.schema import NewsFingerprint
from agent2.schema import TradingSignal
from backtest.backtester import flatten_article_result


def _make_row() -> dict:
    return {
        "ticker": "AAPL",
        "start_date": "2026-05-01",
        "end_date": "2026-05-08",
        "article_text": "Apple beats earnings expectations and raises guidance.",
        "fingpt_label": "up",
        "pass_reason": "",
    }


def _make_fingerprint() -> NewsFingerprint:
    return NewsFingerprint(
        ticker="AAPL",
        source="Reuters",
        published_at="2026-05-01T12:00:00Z",
        headline="Apple beats expectations",
        companies_named=["AAPL"],
        event_keywords=["earnings"],
        sentiment_label="POSITIVE",
        sentiment_score=1.0,
        sentiment_confidence=0.8,
        sentiment_probabilities={
            "POSITIVE": 0.8,
            "NEGATIVE": 0.1,
            "NEUTRAL": 0.1,
        },
        calibration_T=1.3,
        sentiment_logits=[-0.2, -2.0, -2.1],
        event_type="EARNINGS",
        event_type_confidence=0.7,
        event_type_margin=0.4,
        event_type_method="logits_accepted",
        event_type_logits={
            "A": -0.1,
            "B": -3.0,
            "C": -3.1,
            "D": -3.2,
            "E": -3.3,
            "F": -3.4,
            "G": -3.5,
        },
        event_type_probabilities={
            "A": 0.7,
            "B": 0.05,
            "C": 0.05,
            "D": 0.05,
            "E": 0.05,
            "F": 0.05,
            "G": 0.05,
        },
        article_text="Apple beats earnings expectations and raises guidance.",
    )


def _make_signal() -> TradingSignal:
    return TradingSignal(
        ticker="AAPL",
        direction="long",
        strategy_tag="event_driven",
        confidence=0.71,
        raw_signal_logits=[-0.3, -1.0, -2.0],
        pmi_null_logprobs=[-1.2, -1.1, -1.3],
        pmi_alpha_used=0.6,
        signal_logits=[0.42, -0.34, -1.22],
        signal_probabilities={
            "BUY": 0.71,
            "HOLD": 0.2,
            "SELL": 0.09,
        },
        calibration_T=1.3,
        signal_filter_forced_hold=False,
        signal_filter_reason=None,
    )


def test_flatten_article_result_exports_required_numeric_columns():
    row = flatten_article_result(
        _make_row(),
        fingerprint=_make_fingerprint(),
        signal=_make_signal(),
    )

    required_columns = [
        "ticker",
        "fingerprint_ticker",
        "signal_ticker",
        "sentiment_label",
        "sentiment_score",
        "sentiment_confidence",
        "sentiment_logprob_POSITIVE",
        "sentiment_logprob_NEGATIVE",
        "sentiment_logprob_NEUTRAL",
        "sentiment_prob_POSITIVE",
        "sentiment_prob_NEGATIVE",
        "sentiment_prob_NEUTRAL",
        "event_type",
        "event_type_confidence",
        "event_type_margin",
        "event_type_method",
        "event_logprob_A",
        "event_logprob_B",
        "event_logprob_C",
        "event_logprob_D",
        "event_logprob_E",
        "event_logprob_F",
        "event_logprob_G",
        "event_prob_A",
        "event_prob_B",
        "event_prob_C",
        "event_prob_D",
        "event_prob_E",
        "event_prob_F",
        "event_prob_G",
        "direction",
        "confidence",
        "raw_signal_logprob_A",
        "raw_signal_logprob_B",
        "raw_signal_logprob_C",
        "pmi_null_logprob_A",
        "pmi_null_logprob_B",
        "pmi_null_logprob_C",
        "pmi_adjusted_logit_A",
        "pmi_adjusted_logit_B",
        "pmi_adjusted_logit_C",
        "signal_prob_A",
        "signal_prob_B",
        "signal_prob_C",
        "pmi_alpha_used",
        "calibration_T",
        "signal_filter_forced_hold",
        "signal_filter_reason",
        "pass_reason",
    ]

    for column in required_columns:
        assert column in row

    assert row["sentiment_logprob_POSITIVE"] == -0.2
    assert row["fingerprint_ticker"] == "AAPL"
    assert row["signal_ticker"] == "AAPL"
    assert row["event_logprob_A"] == -0.1
    assert row["raw_signal_logprob_A"] == -0.3
    assert row["pmi_null_logprob_A"] == -1.2
    assert row["pmi_adjusted_logit_A"] == 0.42
    assert row["signal_prob_A"] == 0.71
    assert row["pmi_alpha_used"] == 0.6
    assert row["calibration_T"] == 1.3


def test_flatten_article_result_uses_nan_for_malformed_numeric_fields():
    fingerprint = _make_fingerprint().model_copy(
        update={
            "sentiment_logits": "bad",
            "sentiment_probabilities": "bad",
            "event_type_logits": "bad",
            "event_type_probabilities": "bad",
        }
    )
    signal = _make_signal().model_copy(
        update={
            "raw_signal_logits": "bad",
            "pmi_null_logprobs": "bad",
            "signal_logits": "bad",
            "signal_probabilities": "bad",
        }
    )

    row = flatten_article_result(_make_row(), fingerprint=fingerprint, signal=signal)

    assert math.isnan(row["sentiment_logprob_POSITIVE"])
    assert math.isnan(row["sentiment_prob_POSITIVE"])
    assert math.isnan(row["event_logprob_A"])
    assert math.isnan(row["event_prob_A"])
    assert math.isnan(row["raw_signal_logprob_A"])
    assert math.isnan(row["pmi_null_logprob_A"])
    assert math.isnan(row["pmi_adjusted_logit_A"])
    assert math.isnan(row["signal_prob_A"])
