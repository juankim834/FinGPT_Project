"""
agent1/extractor.py — Agent 1: FinGPT fact extractor + logits-based sentiment + event_type.

Pipeline per article
--------------------
1. Fact extraction  — guided-decoding vLLM call → JSON (source, companies, …).
2. Sentiment scoring — direct vLLM scoring → real token logprobs for
                       [POSITIVE, NEGATIVE, NEUTRAL].
3. Event type scoring — direct scoring vLLM call → real token logprobs for
                       [A, B, C, D, E, F, G] mapped to concrete event categories.
4. Deterministic post-processing — softmax(logits / CALIBRATION_T) →
                       probabilities, confidence, label; threshold-based
                       abstention assigns OTHER when confidence or margin is low.
5. Return a NewsFingerprint that combines facts + calibrated sentiment +
   calibrated event_type.

The LLM never outputs a final sentiment label, probability, or confidence
value — all of those are computed in Python from the real logits.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from hashlib import md5
from typing import Any, Optional

from config import (
    CALIBRATION_T,
    EVENT_TYPE_CLASSES,
    EVENT_TYPE_MAP,
    FINGPT_EVENT_TYPE_MIN_CONFIDENCE,
    FINGPT_EVENT_TYPE_MIN_MARGIN,
    FINGPT_MODEL_PATH,
    LOG_LEVEL,
    LOGITS_MAX_TOKENS,
    SENTIMENT_CLASSES,
)
from agent1.prompt import (
    EVENT_TYPE_DECISION_PREFIX,
    EVENT_TYPE_PROMPT,
    EXTRACTION_PROMPT,
    SENTIMENT_PROMPT,
    SENTIMENT_DECISION_PREFIX,
)
from agent1.schema import NewsFingerprint, SentimentLabel, _SENTIMENT_SCORE
from vllm_logits_client import (
    get_real_choice_logits,
    get_real_choice_logits_batch,
    softmax,
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_DIAG_MD_DIR_ENV = "FINGPT_DIAG_MD_DIR"
_SPECIAL_TOKEN_RE = re.compile(
    r"(<\|[^>]+\|>|<｜[^｜]+｜>|<s>|</s>|\[INST\]|\[/INST\])"
)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

_vllm_engine = None
_chat_tokenizer = None


# ---------------------------------------------------------------------------
# Engine management (unchanged public API)
# ---------------------------------------------------------------------------

def _ensure_chat_tokenizer() -> None:
    global _chat_tokenizer  # noqa: PLW0603
    if _chat_tokenizer is not None:
        return
    if not FINGPT_MODEL_PATH:
        return
    try:
        from transformers import AutoTokenizer  # type: ignore

        _chat_tokenizer = AutoTokenizer.from_pretrained(
            FINGPT_MODEL_PATH,
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.warning("Failed to initialize chat tokenizer: %s", exc)


def _load_model() -> None:
    """Lazy-load one vLLM engine once."""
    global _vllm_engine, _chat_tokenizer  # noqa: PLW0603

    if _vllm_engine is not None:
        _ensure_chat_tokenizer()
        return

    if not FINGPT_MODEL_PATH:
        raise EnvironmentError(
            "FINGPT_MODEL_PATH is not set. Add it to your .env pointing to local weights."
        )

    from vllm import LLM  # type: ignore

    logger.info("Loading FinGPT model from: %s", FINGPT_MODEL_PATH)
    _ensure_chat_tokenizer()
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
    """Backward-compatibility shim — vLLM path returns (None, None)."""
    return None, None


def set_shared_vllm_engine(engine) -> None:
    """Inject a preloaded vLLM engine (e.g. from notebook bootstrap)."""
    global _vllm_engine  # noqa: PLW0603
    _vllm_engine = engine
    _ensure_chat_tokenizer()
    logger.info("Injected shared vLLM engine into Agent 1 extractor.")


def get_shared_vllm_engine():
    """Return Agent 1 vLLM engine if already initialized/injected."""
    return _vllm_engine


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_chat_prompt(system_prompt: str, user_content: str) -> str:
    """Format using tokenizer chat template, with a safe fallback."""
    _ensure_chat_tokenizer()
    if _chat_tokenizer is not None and hasattr(_chat_tokenizer, "apply_chat_template"):
        try:
            if system_prompt.strip():
                user_content = f"{system_prompt}\n\n{user_content}"
            messages = [{"role": "user", "content": user_content}]
            return _chat_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.warning("apply_chat_template failed, using fallback prompt: %s", exc)

    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_content}\n"
        "<|assistant|>\n"
    )


# ---------------------------------------------------------------------------
# Text-normalisation helpers (for fact-extraction output)
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _normalize_generated_text(text: str) -> str:
    cleaned = text.replace("Ġ", " ").replace("Ċ", "\n")
    cleaned = _SPECIAL_TOKEN_RE.sub(" ", cleaned)
    cleaned = _THINK_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    marker_match = re.search(r"(?im)^###\s*facts\b", cleaned)
    if marker_match:
        cleaned = cleaned[marker_match.start():].strip()
    return cleaned


def _coerce_string_field(value: Any) -> str:
    """
    Normalize a loosely-typed extracted field into a single string.

    Multi-news samples can cause the extractor to return lists for scalar
    metadata fields such as ``source``. For those cases we keep the first
    non-empty value so the row can still proceed through backtesting.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            normalized = _coerce_string_field(item)
            if normalized:
                return normalized
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _extract_json_blob(text: str) -> str:
    cleaned = _strip_code_fences(text)
    match = _JSON_OBJ_RE.search(cleaned)
    return match.group(0).strip() if match else cleaned


def _extract_balanced_json_blob(text: str) -> str:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1].strip()
    return ""


def _parse_extracted_json(raw_output: str) -> dict[str, Any]:
    primary = _extract_json_blob(raw_output)
    try:
        return json.loads(primary)
    except json.JSONDecodeError:
        balanced = _extract_balanced_json_blob(raw_output)
        if balanced:
            return json.loads(balanced)
        raise


def _parse_markdown_extraction(raw_output: str) -> dict[str, Any]:
    text = _strip_code_fences(raw_output)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    result: dict[str, Any] = {
        "source": "",
        "published_at": "",
        "headline": "",
        "companies_named": [],
        "event_keywords": [],
    }

    def _add_list_items(field: str, raw: str) -> None:
        item = raw.strip().strip("-").strip()
        if not item:
            return
        item = item.replace("**", "").strip()
        if item.lower() == "none":
            return
        if "," in item:
            for part in item.split(","):
                val = part.strip()
                if val and val.lower() != "none":
                    result[field].append(val)
            return
        result[field].append(item)

    active_list: str | None = None
    for line in lines:
        striped = line.strip()
        lower = striped.lower()

        if lower.startswith("### "):
            active_list = None
            continue

        normalized = striped.replace("**", "")
        if normalized.startswith("-"):
            normalized = normalized[1:].strip()
        normalized_lower = normalized.lower()

        if normalized_lower.startswith("source:"):
            result["source"] = normalized.split(":", 1)[1].strip()
            active_list = None
            continue
        if normalized_lower.startswith("published_at:"):
            result["published_at"] = normalized.split(":", 1)[1].strip()
            active_list = None
            continue
        if normalized_lower.startswith("headline:"):
            result["headline"] = normalized.split(":", 1)[1].strip()
            active_list = None
            continue
        if normalized_lower.startswith("companies_named:"):
            active_list = "companies_named"
            remainder = normalized.split(":", 1)[1].strip()
            if remainder:
                _add_list_items(active_list, remainder)
            continue
        if normalized_lower.startswith("event_keywords:"):
            active_list = "event_keywords"
            remainder = normalized.split(":", 1)[1].strip()
            if remainder:
                _add_list_items(active_list, remainder)
            continue

        if active_list is not None and (striped.startswith("-") or striped.startswith("*")):
            item = striped[1:].strip()
            _add_list_items(active_list, item)

    return result


def _parse_extraction_structured(raw_output: str) -> dict[str, Any]:
    if not raw_output.strip():
        raise ValueError("Model returned empty output after normalization")
    try:
        return _parse_extracted_json(raw_output)
    except Exception:
        parsed = _parse_markdown_extraction(raw_output)
        if parsed.get("headline") or parsed.get("companies_named"):
            return parsed
        raise


def _empty_extraction_payload() -> dict[str, Any]:
    """Fallback extraction payload used when fact-extraction output is unusable."""
    return {
        "source": "",
        "published_at": "",
        "headline": "",
        "companies_named": [],
        "event_keywords": [],
    }


def _parse_extraction_or_fallback(
    clean_extraction: str,
    *,
    context_label: str,
) -> dict[str, Any]:
    """
    Parse the fact-extraction output, degrading gracefully on malformed JSON.

    When the extractor output is empty or unparsable, return an empty payload so
    sentiment/event_type can still flow through the backtest using upstream
    ticker/headline fallbacks.
    """
    try:
        return _parse_extraction_structured(clean_extraction)
    except Exception as exc:
        logger.warning(
            "%s fact extraction parse failed (%s); using empty extraction payload.",
            context_label,
            exc,
        )
        return _empty_extraction_payload()


# ---------------------------------------------------------------------------
# Fact-extraction vLLM call (guided decoding, unchanged mechanism)
# ---------------------------------------------------------------------------

def _import_structured_output_params():
    try:
        from vllm.sampling_params import StructuredOutputParams  # type: ignore
        return StructuredOutputParams
    except ImportError:
        logger.warning("StructuredOutputParams not available in current vLLM version.")
    return None


def _build_guided_extraction_schema() -> dict[str, Any]:
    """Return a JSON schema for the extraction-only fields (no sentiment)."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "published_at": {"type": "string"},
            "companies_named": {"type": "array", "items": {"type": "string"}},
            "event_keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["source", "published_at", "companies_named", "event_keywords"],
    }
    return schema


def _generate_extraction_text(
    prompt: str,
    max_tokens: int,
    guided_schema: Optional[dict[str, Any]] = None,
) -> str:
    from vllm import SamplingParams  # type: ignore

    params_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if guided_schema is not None:
        StructuredOutputParams = _import_structured_output_params()
        if StructuredOutputParams is not None:
            params_kwargs["guided_decoding"] = StructuredOutputParams(json=guided_schema)
        else:
            logger.warning(
                "Structured output params unavailable; falling back to non-guided extraction."
            )
    params = SamplingParams(**params_kwargs)
    outputs = _vllm_engine.generate([prompt], params)
    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""


# ---------------------------------------------------------------------------
# Diagnostic / debug helpers
# ---------------------------------------------------------------------------

def _save_md_debug_output(
    article_text: str,
    raw_output: str,
    label: str,
    attempt_idx: int,
    token_budget: int,
) -> None:
    try:
        base_dir = os.getenv(_DIAG_MD_DIR_ENV, "output/diagnostics_md")
        out_dir = os.path.join(base_dir, "agent1")
        os.makedirs(out_dir, exist_ok=True)

        article_id = md5(article_text.encode("utf-8")).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = (
            f"{timestamp}_a1_{article_id}_{label}_attempt{attempt_idx}_tok{token_budget}.md"
        )
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(raw_output)
        logger.info("Saved Agent 1 debug output to %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save Agent 1 debug output: %s", exc)


# ---------------------------------------------------------------------------
# Shared classification helpers (used by both sentiment and event_type)
# ---------------------------------------------------------------------------

def _rank_probabilities(
    probs: list[float],
    labels: list[str],
) -> tuple[int, int, float, float, float]:
    """
    Rank calibrated probabilities and return key statistics.

    Returns
    -------
    (top_idx, second_idx, top_prob, second_prob, margin)
        top_idx    — index of the highest-probability class.
        second_idx — index of the second-highest class.
        top_prob   — probability of the top class.
        second_prob — probability of the second class.
        margin     — top_prob − second_prob.
    """
    if len(probs) < 2:
        top_idx = 0
        return top_idx, 0, probs[0] if probs else 0.0, 0.0, probs[0] if probs else 0.0

    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    top_idx, top_prob = indexed[0]
    second_idx, second_prob = indexed[1]
    margin = top_prob - second_prob
    return top_idx, second_idx, top_prob, second_prob, margin


def _apply_classification_thresholds(
    top_token: str,
    top_prob: float,
    margin: float,
    min_confidence: float,
    min_margin: float,
    label_map: dict[str, str],
) -> tuple[str, str]:
    """
    Apply confidence and margin thresholds to decide whether to accept a
    concrete label or abstain to OTHER.

    Returns
    -------
    (event_type, event_type_method)
    """
    if top_prob < min_confidence:
        return "OTHER", "abstained_low_confidence"
    if margin < min_margin:
        return "OTHER", "abstained_low_margin"
    return label_map[top_token], "logits_accepted"


# ---------------------------------------------------------------------------
# Sentiment post-processing (shared by single-item and batch paths)
# ---------------------------------------------------------------------------

def _process_sentiment_result(
    result: dict[str, Any],
    article_text: str,
    *,
    save_debug: bool = True,
) -> dict[str, Any]:
    """
    Convert a ``get_choice_logits`` / ``get_choice_logits_batch`` result dict
    into the sentiment sub-dict used when building a ``NewsFingerprint``.

    This function contains **all** output logic: softmax, argmax, confidence,
    and the uniform-NEUTRAL fallback.  It is called identically by the
    single-item path (``_score_sentiment``) and the batch path
    (``extract_fingerprint_batch``).
    """
    if save_debug:
        _save_md_debug_output(
            article_text,
            result["raw_output"],
            "sentiment",
            1,
            LOGITS_MAX_TOKENS,
        )

    if result["parse_success"] and result["logits"] is not None:
        logits: list[float] = result["logits"]
        probs = softmax(logits, temperature=CALIBRATION_T)
        best_idx = probs.index(max(probs))
        label: SentimentLabel = SENTIMENT_CLASSES[best_idx]  # type: ignore[assignment]
        confidence = probs[best_idx]
        probabilities = dict(zip(SENTIMENT_CLASSES, probs))
        logger.info(
            "Sentiment logits=%s  probs=%s  label=%s  conf=%.4f",
            logits,
            [f"{p:.4f}" for p in probs],
            label,
            confidence,
        )
    else:
        logger.warning(
            "Sentiment logits parse failed; falling back to NEUTRAL with uniform probs."
        )
        logits = [0.0, 0.0, 0.0]
        probs = [1.0 / 3] * 3
        label = "NEUTRAL"
        confidence = 1.0 / 3
        probabilities = dict(zip(SENTIMENT_CLASSES, probs))

    return {
        "sentiment_label": label,
        "sentiment_score": float(_SENTIMENT_SCORE[label]),
        "sentiment_confidence": round(confidence, 6),
        "sentiment_probabilities": {k: round(v, 6) for k, v in probabilities.items()},
        "sentiment_logits": logits,
        "calibration_T": CALIBRATION_T,
    }


# ---------------------------------------------------------------------------
# Event type post-processing (shared by single-item and batch paths)
# ---------------------------------------------------------------------------

def _process_event_type_result(
    result: dict[str, Any],
    article_text: str,
    *,
    save_debug: bool = True,
    min_confidence: float = FINGPT_EVENT_TYPE_MIN_CONFIDENCE,
    min_margin: float = FINGPT_EVENT_TYPE_MIN_MARGIN,
) -> dict[str, Any]:
    """
    Convert a ``get_real_choice_logits`` result dict into the event_type
    sub-dict used when building a ``NewsFingerprint``.

    Tokens A-G are scored; OTHER is assigned by Python when the confidence
    or margin thresholds are not met.  Designed to be called identically by
    the single-item path and the batch path.
    """
    if save_debug:
        _save_md_debug_output(
            article_text,
            result["raw_output"],
            "event_type",
            1,
            LOGITS_MAX_TOKENS,
        )

    if result["parse_success"] and result["logits"] is not None:
        logits: list[float] = result["logits"]
        probs = softmax(logits, temperature=CALIBRATION_T)
        top_idx, second_idx, top_prob, second_prob, margin = _rank_probabilities(
            probs, EVENT_TYPE_CLASSES
        )
        top_token = EVENT_TYPE_CLASSES[top_idx]
        second_token = EVENT_TYPE_CLASSES[second_idx]

        event_type, method = _apply_classification_thresholds(
            top_token, top_prob, margin, min_confidence, min_margin, EVENT_TYPE_MAP
        )
        secondary_event_type = EVENT_TYPE_MAP[second_token]
        secondary_conf = round(second_prob, 6)

        logits_dict = dict(zip(EVENT_TYPE_CLASSES, logits))
        probs_dict = {k: round(v, 6) for k, v in zip(EVENT_TYPE_CLASSES, probs)}

        logger.info(
            "Event type logits=%s  probs=%s  top=%s(%s)  conf=%.4f  margin=%.4f  method=%s",
            [f"{l:.4f}" for l in logits],
            [f"{p:.4f}" for p in probs],
            top_token,
            event_type,
            top_prob,
            margin,
            method,
        )
    else:
        logger.warning(
            "Event type logits parse failed; assigning OTHER."
        )
        event_type = "OTHER"
        method = "event_type_logits_failed"
        logits_dict = None
        probs_dict = None
        top_prob = None
        margin = None
        secondary_event_type = None
        secondary_conf = None

    return {
        "event_type": event_type,
        "event_type_confidence": round(top_prob, 6) if top_prob is not None else None,
        "event_type_margin": round(margin, 6) if margin is not None else None,
        "event_type_method": method,
        "event_type_logits": logits_dict,
        "event_type_probabilities": probs_dict,
        "secondary_event_type": secondary_event_type,
        "secondary_event_type_confidence": secondary_conf,
    }


# ---------------------------------------------------------------------------
# Sentiment scoring — logits-based (single-item)
# ---------------------------------------------------------------------------

def _score_sentiment(article_text: str) -> dict[str, Any]:
    """
    Extract real model log-probabilities for each sentiment class.

    Uses direct prompt_logprobs scoring of POSITIVE / NEGATIVE / NEUTRAL
    without a chain-of-thought generation phase.

    Output logic is fully delegated to ``_process_sentiment_result``.
    """
    _ensure_chat_tokenizer()
    if _chat_tokenizer is None:
        raise RuntimeError(
            "Tokenizer not loaded — cannot compute real choice logprobs. "
            "Ensure FINGPT_MODEL_PATH is set before calling extract_fingerprint."
        )

    prompt_text = SENTIMENT_PROMPT.format(article_text=article_text)
    prompt = _format_chat_prompt("", prompt_text)

    result = get_real_choice_logits(
        engine=_vllm_engine,
        cot_prompt=prompt,
        choices=SENTIMENT_CLASSES,
        decision_prefix=SENTIMENT_DECISION_PREFIX,
        tokenizer=_chat_tokenizer,
        max_cot_tokens=0,
        use_cot=False,
    )
    return _process_sentiment_result(result, article_text)


# ---------------------------------------------------------------------------
# Event type scoring — logits-based (single-item)
# ---------------------------------------------------------------------------

def _score_event_type(article_text: str) -> dict[str, Any]:
    """
    Extract real model log-probabilities for each of the 7 event type tokens.

    Uses direct prompt_logprobs scoring (no CoT) — the event_type prompt is
    concise and self-contained.  Score tokens are A-G; OTHER is never scored
    by the model and is assigned by Python via threshold logic.

    Output logic is fully delegated to ``_process_event_type_result``.
    """
    _ensure_chat_tokenizer()
    if _chat_tokenizer is None:
        raise RuntimeError(
            "Tokenizer not loaded — cannot compute real choice logprobs. "
            "Ensure FINGPT_MODEL_PATH is set before calling extract_fingerprint."
        )

    event_prompt_text = EVENT_TYPE_PROMPT.format(article_text=article_text)
    prompt = _format_chat_prompt("", event_prompt_text)

    result = get_real_choice_logits(
        engine=_vllm_engine,
        cot_prompt=prompt,
        choices=EVENT_TYPE_CLASSES,
        decision_prefix=EVENT_TYPE_DECISION_PREFIX,
        tokenizer=_chat_tokenizer,
        max_cot_tokens=0,   # no CoT for event_type — score directly
        use_cot=False,
    )
    return _process_event_type_result(result, article_text)


# ---------------------------------------------------------------------------
# Fingerprint assembly (shared by single-item and batch paths)
# ---------------------------------------------------------------------------

def _assemble_fingerprint(
    extracted: dict[str, Any],
    sentiment: dict[str, Any],
    event_type: dict[str, Any],
    article_text: str,
    fallback_ticker: Optional[str] = None,
    fallback_headline: Optional[str] = None,
) -> NewsFingerprint:
    """
    Build a ``NewsFingerprint`` from fact-extraction, sentiment, and event_type dicts.

    ``ticker`` is carried separately from the extracted company mentions.
    If ``companies_named`` is empty and ``fallback_ticker`` is provided,
    the ticker is also substituted into ``companies_named`` to avoid
    dropping otherwise valid rows.
    ``event_keywords`` is set to ``[event_type]`` for backward compatibility.
    """
    companies_named = extracted.get("companies_named", [])
    if not companies_named and fallback_ticker:
        companies_named = [fallback_ticker]
        logger.info(
            "companies_named empty; using dataset ticker fallback: %s", fallback_ticker
        )

    et = event_type.get("event_type", "OTHER")
    payload: dict[str, Any] = {
        "ticker": fallback_ticker or "",
        "source": _coerce_string_field(extracted.get("source", "")),
        "published_at": _coerce_string_field(extracted.get("published_at", "")),
        "headline": fallback_headline or extracted.get("headline", ""),
        "companies_named": companies_named,
        # Backward compat: set to [event_type] so downstream code that reads
        # event_keywords still gets a meaningful single-item list.
        "event_keywords": [et.lower()],
        "sentiment_label": sentiment["sentiment_label"],
        "sentiment_score": sentiment["sentiment_score"],
        "sentiment_confidence": sentiment["sentiment_confidence"],
        "sentiment_probabilities": sentiment["sentiment_probabilities"],
        "sentiment_logits": sentiment["sentiment_logits"],
        "calibration_T": sentiment["calibration_T"],
        "event_type": et,
        "event_type_confidence": event_type.get("event_type_confidence"),
        "event_type_margin": event_type.get("event_type_margin"),
        "event_type_method": event_type.get("event_type_method"),
        "event_type_logits": event_type.get("event_type_logits"),
        "event_type_probabilities": event_type.get("event_type_probabilities"),
        "secondary_event_type": event_type.get("secondary_event_type"),
        "secondary_event_type_confidence": event_type.get("secondary_event_type_confidence"),
        "article_text": article_text,
    }
    return NewsFingerprint(**payload)


def _build_extraction_prompt(article_text: str) -> str:
    user_content = f"{EXTRACTION_PROMPT}\n\nArticle:\n{article_text}"
    return _format_chat_prompt("", user_content)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def extract_fingerprint_batch(
    article_texts: list[str],
    tickers: Optional[list[str]] = None,
    headlines: Optional[list[str]] = None,
) -> list[Optional[NewsFingerprint]]:
    """
    Batch version of ``extract_fingerprint``.

    Sends all articles to vLLM in three batched inference groups:
      1. Guided fact-extraction call (all articles, single ``SamplingParams``).
      2. Sentiment direct logits call (all articles, single ``SamplingParams``).
      3. Event type direct logits call (all articles, single ``SamplingParams``).

    Per-item output logic (softmax, argmax, ``NewsFingerprint`` assembly) is
    identical to the single-item path — handled by ``_process_sentiment_result``,
    ``_process_event_type_result``, and ``_assemble_fingerprint``.

    Parameters
    ----------
    article_texts:
        Raw news article strings.
    tickers:
        Optional list of tickers (same length as article_texts) used as
        fallback when companies_named is empty (dataset ticker fallback).
    headlines:
        Optional list of raw headlines (same length as article_texts). When
        provided, the fingerprint keeps this external headline instead of a
        model-generated one.

    Returns a list of the same length as *article_texts*.  Items are ``None``
    wherever extraction or validation fails.
    """
    _load_model()
    _ensure_chat_tokenizer()

    if _chat_tokenizer is None:
        raise RuntimeError(
            "Tokenizer not loaded — cannot compute real choice logprobs. "
            "Ensure FINGPT_MODEL_PATH is set before calling extract_fingerprint_batch."
        )

    n = len(article_texts)
    if n == 0:
        return []

    if tickers is not None and len(tickers) != n:
        logger.warning(
            "tickers length (%d) != article_texts length (%d); ignoring tickers.",
            len(tickers), n,
        )
        tickers = None
    if headlines is not None and len(headlines) != n:
        logger.warning(
            "headlines length (%d) != article_texts length (%d); ignoring headlines.",
            len(headlines), n,
        )
        headlines = None

    from vllm import SamplingParams  # type: ignore

    # --- Step 1: Batch guided fact extraction (1 vLLM call) ---
    extraction_prompts = [_build_extraction_prompt(t) for t in article_texts]
    guided_schema = _build_guided_extraction_schema()
    ext_params_kwargs: dict[str, Any] = {"max_tokens": 2048, "temperature": 0.0}
    StructuredOutputParams = _import_structured_output_params()
    if StructuredOutputParams is not None:
        ext_params_kwargs["guided_decoding"] = StructuredOutputParams(json=guided_schema)
    else:
        logger.warning(
            "StructuredOutputParams unavailable; batch extraction runs without guided decoding."
        )
    ext_params = SamplingParams(**ext_params_kwargs)
    ext_outputs = _vllm_engine.generate(extraction_prompts, ext_params)
    raw_extractions = [
        o.outputs[0].text if o.outputs else "" for o in ext_outputs
    ]

    # --- Step 2: Batch real logprobs for sentiment (single scoring call) ---
    sentiment_prompts = [
        _format_chat_prompt("", SENTIMENT_PROMPT.format(article_text=t))
        for t in article_texts
    ]
    sentiment_results = get_real_choice_logits_batch(
        engine=_vllm_engine,
        cot_prompts=sentiment_prompts,
        choices=SENTIMENT_CLASSES,
        decision_prefix=SENTIMENT_DECISION_PREFIX,
        tokenizer=_chat_tokenizer,
        max_cot_tokens=0,
        use_cot=False,
    )

    # --- Step 3: Batch real logprobs for event_type (1 vLLM call — no CoT) ---
    event_type_prompts = [
        _format_chat_prompt("", EVENT_TYPE_PROMPT.format(article_text=t))
        for t in article_texts
    ]
    event_type_results = get_real_choice_logits_batch(
        engine=_vllm_engine,
        cot_prompts=event_type_prompts,
        choices=EVENT_TYPE_CLASSES,
        decision_prefix=EVENT_TYPE_DECISION_PREFIX,
        tokenizer=_chat_tokenizer,
        max_cot_tokens=0,
        use_cot=False,
    )

    # --- Step 4: Per-item assembly (output logic unchanged) ---
    fingerprints: list[Optional[NewsFingerprint]] = []
    for i, article_text in enumerate(article_texts):
        fallback_ticker = tickers[i] if tickers else None
        fallback_headline = headlines[i] if headlines else None
        try:
            clean_extraction = _normalize_generated_text(raw_extractions[i])
            _save_md_debug_output(article_text, clean_extraction, "extraction", 1, 2048)
            logger.info(
                "Batch extraction[%d] output (first 240 chars): %s",
                i, clean_extraction[:240],
            )
            extracted = _parse_extraction_or_fallback(
                clean_extraction,
                context_label=f"extract_fingerprint_batch[{i}]",
            )

            sentiment = _process_sentiment_result(
                sentiment_results[i], article_text, save_debug=True
            )
            event_type = _process_event_type_result(
                event_type_results[i], article_text, save_debug=True
            )

            fingerprints.append(
                _assemble_fingerprint(
                    extracted,
                    sentiment,
                    event_type,
                    article_text,
                    fallback_ticker,
                    fallback_headline,
                )
            )
        except Exception as exc:
            logger.error(
                "extract_fingerprint_batch[%d] failed: %s", i, exc, exc_info=True
            )
            fingerprints.append(None)

    return fingerprints


def extract_fingerprint(
    article_text: str,
    ticker: Optional[str] = None,
    headline: Optional[str] = None,
) -> Optional[NewsFingerprint]:
    """
    Extract structured facts, logits-based sentiment, and logits-based event_type
    from a raw news article.

    Parameters
    ----------
    article_text:
        Raw news article text.
    ticker:
        Optional ticker used as companies_named fallback when the model
        fails to extract any company name.
    headline:
        Optional raw headline preserved in the fingerprint. When omitted,
        Agent 1 may fall back to the extracted headline field.

    Returns None on any model, parse, or validation failure.
    """
    try:
        _load_model()

        # Fact extraction (guided decoding).
        extraction_prompt = _build_extraction_prompt(article_text)
        guided_schema = _build_guided_extraction_schema()
        token_budget = 2048
        raw_extraction = _generate_extraction_text(
            extraction_prompt,
            max_tokens=token_budget,
            guided_schema=guided_schema,
        )
        clean_extraction = _normalize_generated_text(raw_extraction)
        _save_md_debug_output(article_text, clean_extraction, "extraction", 1, token_budget)
        logger.info(
            "Agent 1 fact extraction output (max_tokens=%d): %s",
            token_budget,
            clean_extraction[:240],
        )
        extracted = _parse_extraction_or_fallback(
            clean_extraction,
            context_label="extract_fingerprint",
        )

        # Logits-based sentiment scoring.
        sentiment = _score_sentiment(article_text)

        # Logits-based event_type scoring.
        event_type = _score_event_type(article_text)

        return _assemble_fingerprint(
            extracted,
            sentiment,
            event_type,
            article_text,
            ticker,
            headline,
        )

    except Exception as exc:
        logger.error("extract_fingerprint failed: %s", exc, exc_info=True)
        return None
