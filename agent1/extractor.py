"""
agent1/extractor.py — Agent 1: FinGPT fact + sentiment extractor.

The module keeps a lazy, single-load vLLM setup and exposes one public
API: extract_fingerprint(article_text) -> Optional[NewsFingerprint].
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from hashlib import md5
from typing import Any, Optional

from config import (
    FINGPT_MAX_NEW_TOKENS,
    FINGPT_MODEL_PATH,
    FINGPT_TEMPERATURE,
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

_vllm_engine = None
_chat_tokenizer = None


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

    if _vllm_engine is not None and _chat_tokenizer is not None:
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
            return _chat_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
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


def _extract_json_blob(text: str) -> str:
    cleaned = _strip_code_fences(text)
    match = _JSON_OBJ_RE.search(cleaned)
    return match.group(0).strip() if match else cleaned


def _extract_balanced_json_blob(text: str) -> str:
    """
    Extract first balanced JSON object from text with string-aware brace matching.
    Returns empty string if no balanced object is found.
    """
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
    """
    Parse extraction JSON robustly from model output.
    """
    primary = _extract_json_blob(raw_output)
    try:
        return json.loads(primary)
    except json.JSONDecodeError:
        balanced = _extract_balanced_json_blob(raw_output)
        if balanced:
            return json.loads(balanced)
        raise


def _parse_markdown_extraction(raw_output: str) -> dict[str, Any]:
    """
    Parse Agent 1 markdown contract into JSON-like dict.
    """
    text = _strip_code_fences(raw_output)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    result: dict[str, Any] = {
        "source": "",
        "published_at": "",
        "headline": "",
        "companies_named": [],
        "event_keywords": [],
    }

    active_list: str | None = None
    for line in lines:
        striped = line.strip()
        lower = striped.lower()

        if lower.startswith("### "):
            active_list = None
            continue

        if lower.startswith("- source:"):
            result["source"] = striped.split(":", 1)[1].strip()
            active_list = None
            continue
        if lower.startswith("- published_at:"):
            result["published_at"] = striped.split(":", 1)[1].strip()
            active_list = None
            continue
        if lower.startswith("- headline:"):
            result["headline"] = striped.split(":", 1)[1].strip()
            active_list = None
            continue
        if lower.startswith("- companies_named:"):
            active_list = "companies_named"
            continue
        if lower.startswith("- event_keywords:"):
            active_list = "event_keywords"
            continue

        if striped.startswith("- ") and active_list is not None:
            item = striped[2:].strip()
            if item and item.lower() != "none":
                result[active_list].append(item)
            continue

    return result


def _parse_extraction_structured(raw_output: str) -> dict[str, Any]:
    """
    Parse model output that may be JSON or markdown.
    """
    try:
        return _parse_extracted_json(raw_output)
    except Exception:
        parsed = _parse_markdown_extraction(raw_output)
        if parsed.get("headline") or parsed.get("companies_named"):
            return parsed
        raise


def _generate_extraction_text(prompt: str, max_tokens: int) -> str:
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=FINGPT_TEMPERATURE,
    )
    outputs = _vllm_engine.generate([prompt], params)
    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""


def _save_md_debug_output(
    article_text: str,
    raw_output: str,
    attempt_idx: int,
    token_budget: int,
) -> None:
    """
    Persist raw extraction output for debugging in markdown files.
    """
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
    return _format_chat_prompt(AGENT1_SYSTEM_PROMPT, article_text)


def _build_sentiment_prompt(article_text: str) -> str:
    labels = ", ".join(_SENTIMENT_LABELS)
    user_prompt = (
        "Classify the market sentiment of this news article.\n"
        f"Valid labels: {labels}.\n"
        "Respond with only one label.\n\n"
        f"Article:\n{article_text}\n\nLabel:"
    )
    return _format_chat_prompt("You are a financial sentiment classifier.", user_prompt)


def _normalize_sentiment_label(raw_text: str) -> SentimentLabel:
    text = raw_text.strip().lower()
    for label in _SENTIMENT_LABELS:
        if label in text:
            return label
    # Deterministic fallback if model returns unexpected wording.
    return "neutral"


def _score_sentiment(article_text: str) -> dict[str, Any]:
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
        token_budgets = [FINGPT_MAX_NEW_TOKENS, max(FINGPT_MAX_NEW_TOKENS * 2, 1024)]
        extracted: dict[str, Any] | None = None
        last_parse_error: Exception | None = None

        for attempt_idx, token_budget in enumerate(token_budgets, start=1):
            raw_output = _generate_extraction_text(prompt, max_tokens=token_budget)
            _save_md_debug_output(article_text, raw_output, attempt_idx, token_budget)
            logger.info(
                "FinGPT raw extraction output (attempt %d, max_tokens=%d): %s",
                attempt_idx,
                token_budget,
                raw_output[:240],
            )
            try:
                extracted = _parse_extraction_structured(raw_output)
                break
            except json.JSONDecodeError as exc:
                last_parse_error = exc
                logger.warning(
                    "Agent 1 JSON parse failed (attempt %d, max_tokens=%d): %s",
                    attempt_idx,
                    token_budget,
                    exc,
                )
            except Exception as exc:
                last_parse_error = exc
                logger.warning(
                    "Agent 1 structured parse failed (attempt %d, max_tokens=%d): %s",
                    attempt_idx,
                    token_budget,
                    exc,
                )

        if extracted is None:
            raise RuntimeError("Agent 1 failed to produce valid JSON output.") from last_parse_error

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
