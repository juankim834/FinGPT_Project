# Required change: removed GuidedDecodingParams fallback and use StructuredOutputParams only.
"""
agent1/extractor.py — Agent 1: FinGPT fact + sentiment extractor.

The module keeps a lazy, single-load vLLM setup and exposes one public
API: extract_fingerprint(article_text) -> Optional[NewsFingerprint].
"""

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from hashlib import md5
from typing import Any, Optional

from config import (
    FINGPT_MODEL_PATH,
    LOG_LEVEL,
)
from agent1.prompt import SYSTEM_PROMPT as AGENT1_SYSTEM_PROMPT
from agent1.schema import NewsFingerprint, SentimentLabel

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

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
_DIAG_MD_DIR_ENV = "FINGPT_DIAG_MD_DIR"
_SPECIAL_TOKEN_RE = re.compile(
    r"(<\|[^>]+\|>|<｜[^｜]+｜>|<s>|</s>|\[INST\]|\[/INST\])"
)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

_vllm_engine = None
_chat_tokenizer = None


def _import_structured_output_params():
    """
    Import structured output params class (vLLM >= 0.12.0).
    Returns the class or None if unavailable.
    """
    try:
        from vllm.sampling_params import StructuredOutputParams  # type: ignore
        return StructuredOutputParams
    except ImportError:
        logger.warning("StructuredOutputParams not available in current vLLM version.")
    return None


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
    _ensure_chat_tokenizer()
    logger.info("Injected shared vLLM engine into Agent 1 extractor.")


def get_shared_vllm_engine():
    """Return Agent 1 vLLM engine if already initialized/injected."""
    return _vllm_engine


def _format_chat_prompt(system_prompt: str, user_content: str) -> str:
    """
    Format prompt using tokenizer chat template (preferred), with fallback.
    """
    _ensure_chat_tokenizer()
    if _chat_tokenizer is not None and hasattr(_chat_tokenizer, "apply_chat_template"):
        try:
            if system_prompt.strip():
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
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
            continue

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


def _build_guided_extraction_schema() -> dict[str, Any]:
    schema = NewsFingerprint.model_json_schema()
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for field in ("sentiment_label", "sentiment_score", "sentiment_confidence"):
        props.pop(field, None)
        if field in required:
            required.remove(field)
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
                "Structured output params unavailable in current vLLM version; "
                "falling back to non-guided extraction."
            )
    params = SamplingParams(**params_kwargs)
    outputs = _vllm_engine.generate([prompt], params)
    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""


def _save_md_debug_output(
    article_text: str,
    raw_output: str,
    attempt_idx: int,
    token_budget: int,
) -> None:
    try:
        base_dir = os.getenv(_DIAG_MD_DIR_ENV, "output/diagnostics_md")
        out_dir = os.path.join(base_dir, "agent1")
        os.makedirs(out_dir, exist_ok=True)

        article_id = md5(article_text.encode("utf-8")).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp}_a1_{article_id}_attempt{attempt_idx}_tok{token_budget}.md"
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(raw_output)
        logger.info("Saved Agent 1 markdown debug output to %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save Agent 1 markdown debug output: %s", exc)


def _build_extraction_prompt(article_text: str) -> str:
    user_content = f"{AGENT1_SYSTEM_PROMPT}\n\nArticle:\n{article_text}"
    return _format_chat_prompt("", user_content)


def _build_sentiment_prompt(article_text: str) -> str:
    labels = ", ".join(_SENTIMENT_LABELS)
    user_prompt = (
        "Classify the market sentiment of this news article.\n"
        f"Valid labels: {labels}.\n"
        "Respond with only one label.\n\n"
        f"Article:\n{article_text}\n\nLabel:"
    )
    return _format_chat_prompt("", user_prompt)


def _normalize_sentiment_label(raw_text: str) -> SentimentLabel:
    text = raw_text.strip().lower()
    for label in _SENTIMENT_LABELS:
        if label in text:
            return label
    return "neutral"


def _score_sentiment(article_text: str) -> dict[str, Any]:
    from vllm import SamplingParams  # type: ignore

    prompt = _build_sentiment_prompt(article_text)
    label_schema = {"type": "string", "enum": list(_SENTIMENT_LABELS)}

    params_kwargs: dict[str, Any] = {
        "max_tokens": 32,
        "temperature": 0.0,
        "logprobs": len(_SENTIMENT_LABELS),
    }
    structured_available = False
    StructuredOutputParams = _import_structured_output_params()
    if StructuredOutputParams is not None:
        params_kwargs["guided_decoding"] = StructuredOutputParams(json=label_schema)
        structured_available = True
    else:
        logger.warning(
            "Structured output params unavailable in current vLLM version; "
            "falling back to plain sentiment inference."
        )

    params = SamplingParams(**params_kwargs)
    outputs = _vllm_engine.generate([prompt], params)
    raw = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""

    sentiment_label = _normalize_sentiment_label(raw)
    sentiment_score = float(_SENTIMENT_VALUES[sentiment_label])

    token_logprobs = outputs[0].outputs[0].logprobs
    if token_logprobs and structured_available:
        top_logprob = max(token_logprobs[0].values(), key=lambda x: x.logprob).logprob
        sentiment_confidence = round(math.exp(top_logprob), 4)
    else:
        sentiment_confidence = 1.0

    sentiment_logits = {
        label: (1.0 if label == sentiment_label else 0.0) for label in _SENTIMENT_LABELS
    }

    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "sentiment_confidence": sentiment_confidence,
        "sentiment_logits": sentiment_logits,
    }


def extract_fingerprint(article_text: str) -> Optional[NewsFingerprint]:
    """
    Extract structured facts and log-prob sentiment from a raw news article string.

    Returns None on any model, parse, or validation failure.
    """
    try:
        _load_model()

        prompt = _build_extraction_prompt(article_text)
        guided_schema = _build_guided_extraction_schema()
        token_budget = 2048
        raw_output = _generate_extraction_text(
            prompt,
            max_tokens=token_budget,
            guided_schema=guided_schema,
        )
        clean_output = _normalize_generated_text(raw_output)
        _save_md_debug_output(article_text, clean_output, 1, token_budget)
        logger.info(
            "FinGPT raw extraction output (attempt %d, max_tokens=%d): %s",
            1,
            token_budget,
            clean_output[:240],
        )
        extracted = _parse_extraction_structured(clean_output)

        sentiment = _score_sentiment(article_text)
        logger.info("Agent 1 sentiment logits: %s", sentiment["sentiment_logits"])

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