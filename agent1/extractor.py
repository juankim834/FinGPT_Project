"""
agent1/extractor.py — Agent 1: FinGPT fact + sentiment extractor.

The module keeps a lazy, single-load vLLM setup and exposes one public
API: extract_fingerprint(article_text) -> Optional[NewsFingerprint].
"""

import json
import logging
import re
from typing import Optional

from config import (
    FINGPT_MAX_NEW_TOKENS,
    FINGPT_MODEL_PATH,
    FINGPT_TEMPERATURE,
    LOG_LEVEL,
)
from agent1.schema import NewsFingerprint, SentimentLabel

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a financial news fact extractor.\n"
    "Return exactly one JSON object with keys:\n"
    'source, published_at, headline, companies_named, event_keywords\n'
    "Rules:\n"
    "- Only use explicit article facts.\n"
    "- headline must be copied exactly from text when present.\n"
    "- companies_named should include all company names/tickers in the article.\n"
    "- event_keywords must be lowercase short keywords copied from text.\n"
    "- No markdown, no commentary, no extra keys."
)

_SENTIMENT_LABELS: list[SentimentLabel] = [
    "strongly bullish",
    "bullish",
    "neutral",
    "bearish",
    "strongly bearish",
]
_SENTIMENT_VALUES: dict[SentimentLabel, int] = {
    "strongly bullish": 2,
    "bullish": 1,
    "neutral": 0,
    "bearish": -1,
    "strongly bearish": -2,
}

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_vllm_engine = None


def _load_model() -> None:
    """Lazy-load one vLLM engine once."""
    global _vllm_engine  # noqa: PLW0603

    if _vllm_engine is not None:
        return

    if not FINGPT_MODEL_PATH:
        raise EnvironmentError(
            "FINGPT_MODEL_PATH is not set. Add it to your .env pointing to local weights."
        )

    from vllm import LLM  # type: ignore

    logger.info("Loading FinGPT model from: %s", FINGPT_MODEL_PATH)
    _vllm_engine = LLM(
        model=FINGPT_MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
        enforce_eager=True,
    )
    logger.info("FinGPT vLLM engine loaded.")


def get_loaded_model_and_tokenizer():
    """
    Backward compatibility shim for Agent 2 shared-loader path.

    Agent 1 now uses vLLM directly, so no HF model/tokenizer pair is returned.
    """
    return None, None


def set_shared_vllm_engine(engine) -> None:
    """
    Inject a preloaded vLLM engine (e.g., from notebook bootstrap) for reuse.
    """
    global _vllm_engine  # noqa: PLW0603
    _vllm_engine = engine
    logger.info("Injected shared vLLM engine into Agent 1 extractor.")


def get_shared_vllm_engine():
    """Return Agent 1 vLLM engine if already initialized/injected."""
    return _vllm_engine


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _extract_json_blob(text: str) -> str:
    cleaned = _strip_code_fences(text)
    match = _JSON_OBJ_RE.search(cleaned)
    return match.group(0).strip() if match else cleaned


def _build_extraction_prompt(article_text: str) -> str:
    return (
        f"<|system|>\n{_EXTRACTION_SYSTEM_PROMPT}\n"
        f"<|user|>\n{article_text}\n"
        "<|assistant|>\n"
    )


def _build_sentiment_prompt(article_text: str) -> str:
    labels = ", ".join(_SENTIMENT_LABELS)
    return (
        "Classify the market sentiment of this news article.\n"
        f"Valid labels: {labels}.\n"
        "Respond with only one label.\n\n"
        f"Article:\n{article_text}\n\nLabel:"
    )


def _normalize_sentiment_label(raw_text: str) -> SentimentLabel:
    text = raw_text.strip().lower()
    for label in _SENTIMENT_LABELS:
        if label in text:
            return label
    # Deterministic fallback if model returns unexpected wording.
    return "neutral"


def _score_sentiment(article_text: str) -> dict[str, float | str]:
    """
    Classify sentiment using deterministic vLLM decoding over fixed labels.
    """
    from vllm import SamplingParams  # type: ignore

    prompt = _build_sentiment_prompt(article_text)
    params = SamplingParams(max_tokens=6, temperature=0.0)
    outputs = _vllm_engine.generate([prompt], params)
    raw = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""

    sentiment_label = _normalize_sentiment_label(raw)
    sentiment_score = float(_SENTIMENT_VALUES[sentiment_label])
    sentiment_confidence = 1.0

    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "sentiment_confidence": sentiment_confidence,
    }


def extract_fingerprint(article_text: str) -> Optional[NewsFingerprint]:
    """
    Extract structured facts and log-prob sentiment from a raw news article string.

    Returns None on any model, parse, or validation failure.
    """
    try:
        _load_model()
        from vllm import SamplingParams  # type: ignore

        prompt = _build_extraction_prompt(article_text)
        params = SamplingParams(
            max_tokens=FINGPT_MAX_NEW_TOKENS,
            temperature=FINGPT_TEMPERATURE,
        )
        outputs = _vllm_engine.generate([prompt], params)
        raw_output = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        logger.info("FinGPT raw extraction output: %s", raw_output[:240])

        extracted = json.loads(_extract_json_blob(raw_output))
        sentiment = _score_sentiment(article_text)

        payload = {
            "source": extracted.get("source", ""),
            "published_at": extracted.get("published_at", ""),
            "headline": extracted.get("headline", ""),
            "companies_named": extracted.get("companies_named", []),
            "event_keywords": extracted.get("event_keywords", []),
            "sentiment_label": sentiment["sentiment_label"],
            "sentiment_score": sentiment["sentiment_score"],
            "sentiment_confidence": sentiment["sentiment_confidence"],
        }
        return NewsFingerprint(**payload)

    except Exception as exc:
        logger.error("extract_fingerprint failed: %s", exc, exc_info=True)
        return None
