# Required change: switched guided decoding import to StructuredOutputParams for vLLM >= 0.12.0.
"""
agent2/reasoner.py — Agent 2 vLLM reasoner.

Converts a NewsFingerprint into a TradingSignal using a locally loaded
DeepSeek model. Raw reasoning text is persisted to logs/cot_<ticker>_<timestamp>.txt.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from hashlib import md5
from typing import Any, Optional

from config import (
    FINGPT_MODEL_PATH,
    LOG_LEVEL,
    LOGS_DIR,
    SHARE_SINGLE_LLM_BETWEEN_AGENTS,
)
from agent1.schema import NewsFingerprint
from agent2.prompt import SYSTEM_PROMPT
from agent2.schema import TradingSignal

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
    """Lazy-load a vLLM engine once; reuse Agent 1 engine in shared mode."""
    global _vllm_engine, _chat_tokenizer  # noqa: PLW0603

    if _vllm_engine is not None:
        _ensure_chat_tokenizer()
        return

    _ensure_chat_tokenizer()

    if SHARE_SINGLE_LLM_BETWEEN_AGENTS:
        from agent1.extractor import get_shared_vllm_engine

        shared_engine = get_shared_vllm_engine()
        if shared_engine is not None:
            _vllm_engine = shared_engine
            logger.info("Agent 2 reusing shared vLLM engine from Agent 1.")
            return
        logger.info(
            "Shared vLLM engine not yet available from Agent 1; "
            "Agent 2 will initialize its own vLLM engine."
        )

    if not FINGPT_MODEL_PATH:
        raise EnvironmentError(
            "FINGPT_MODEL_PATH is not set. Add it to your .env pointing to local weights."
        )

    from vllm import LLM  # type: ignore

    logger.info("Loading Agent 2 vLLM engine from: %s", FINGPT_MODEL_PATH)
    _ensure_chat_tokenizer()
    _vllm_engine = LLM(
        model=FINGPT_MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
        enforce_eager=True,
    )
    logger.info("Agent 2 vLLM engine loaded.")


def set_shared_vllm_engine(engine) -> None:
    """Inject a preloaded vLLM engine for Agent 2 reuse."""
    global _vllm_engine  # noqa: PLW0603
    _vllm_engine = engine
    _ensure_chat_tokenizer()
    logger.info("Injected shared vLLM engine into Agent 2 reasoner.")


def _format_chat_prompt(system_prompt: str, user_content: str) -> str:
    """
    Format prompt using tokenizer chat template (preferred), with fallback.
    """
    _ensure_chat_tokenizer()
    if _chat_tokenizer is not None and hasattr(_chat_tokenizer, "apply_chat_template"):
        try:
            # Match evaluation behavior: single user message template path.
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


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _normalize_generated_text(text: str) -> str:
    # Decode common tokenized artifacts seen in debug outputs.
    cleaned = text.replace("Ġ", " ").replace("Ċ", "\n")
    cleaned = _SPECIAL_TOKEN_RE.sub(" ", cleaned)
    cleaned = _THINK_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    # Keep only structured signal section if chain-of-thought leaks ahead of it.
    marker_match = re.search(r"(?im)^###\s*signal\b", cleaned)
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


def _parse_json_signal(raw_text: str) -> dict[str, Any]:
    primary = _extract_json_blob(raw_text)
    try:
        return json.loads(primary)
    except json.JSONDecodeError:
        balanced = _extract_balanced_json_blob(raw_text)
        if balanced:
            return json.loads(balanced)
        raise


def _parse_markdown_signal(raw_text: str) -> dict[str, Any]:
    text = _strip_code_fences(raw_text)
    signal_markers = list(re.finditer(r"(?im)^###\s*signal\b", text))
    if signal_markers:
        # Prefer the last complete SIGNAL block when multiple are present.
        text = text[signal_markers[-1].start():]
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    result: dict[str, Any] = {
        "ticker": "",
        "direction": "neutral",
        "strategy_tag": "none",
        "confidence": 0.0,
        "cot": "",
    }

    def _extract_value(line_text: str) -> str:
        normalized = line_text.replace("**", "")
        if normalized.startswith("-"):
            normalized = normalized[1:].strip()
        return normalized.split(":", 1)[1].strip() if ":" in normalized else ""

    cot_started = False
    for line in lines:
        lower = line.lower()
        if lower.startswith("### "):
            continue
        normalized = line.replace("**", "").strip()
        if normalized.startswith("-"):
            normalized = normalized[1:].strip()
        normalized_lower = normalized.lower()

        if normalized_lower.startswith("ticker:"):
            result["ticker"] = _extract_value(line)
            cot_started = False
            continue
        if normalized_lower.startswith("direction:"):
            result["direction"] = _extract_value(line).lower()
            cot_started = False
            continue
        if normalized_lower.startswith("strategy_tag:"):
            result["strategy_tag"] = _extract_value(line).lower()
            cot_started = False
            continue
        if normalized_lower.startswith("confidence:"):
            raw_conf = _extract_value(line)
            raw_conf = raw_conf.replace("%", "").strip()
            try:
                conf = float(raw_conf)
                if conf > 1.0:
                    conf = conf / 100.0
                result["confidence"] = conf
            except ValueError:
                # Keep default and let schema validation catch if needed.
                pass
            cot_started = False
            continue
        if normalized_lower.startswith("cot:"):
            result["cot"] = _extract_value(line)
            cot_started = True
            continue
        if cot_started and not normalized_lower.startswith(("ticker:", "direction:", "strategy_tag:", "confidence:")):
            # Allow wrapped reasoning lines after `- cot:`.
            result["cot"] = f"{result['cot']} {line}".strip()

    # Compact single-line fallback:
    # "### SIGNAL-ticker: X - direction: long - strategy_tag: momentum - confidence: 0.7 - cot: ..."
    if not result["ticker"] or not result["cot"]:
        collapsed = " ".join(lines)
        collapsed = re.sub(r"\s+", " ", collapsed).strip()
        keys = ["ticker", "direction", "strategy_tag", "confidence", "cot"]
        key_union = "|".join(keys)

        def _extract(field: str) -> str:
            pattern = rf"{field}\s*:\s*(.*?)(?=\s+-\s*(?:{key_union})\s*:|$)"
            match = re.search(pattern, collapsed, flags=re.IGNORECASE)
            return match.group(1).strip() if match else ""

        ticker = _extract("ticker")
        direction = _extract("direction").lower()
        strategy_tag = _extract("strategy_tag").lower()
        confidence_raw = _extract("confidence").replace("%", "").strip()
        cot = _extract("cot")

        if ticker:
            result["ticker"] = ticker
        if direction:
            result["direction"] = direction
        if strategy_tag:
            result["strategy_tag"] = strategy_tag
        if confidence_raw:
            try:
                conf = float(confidence_raw)
                if conf > 1.0:
                    conf = conf / 100.0
                result["confidence"] = conf
            except ValueError:
                pass
        if cot:
            result["cot"] = cot

    return result


def _parse_signal_structured(raw_text: str) -> dict[str, Any]:
    try:
        return _parse_json_signal(raw_text)
    except Exception:
        parsed = _parse_markdown_signal(raw_text)
        if parsed.get("ticker") and parsed.get("cot"):
            return parsed
        raise


def _parse_signal_fallback(raw_text: str, fingerprint: NewsFingerprint) -> dict[str, Any]:
    """
    Last-resort parser for loosely formatted model outputs.
    """
    text = raw_text.strip()
    lower = text.lower()

    ticker = fingerprint.companies_named[0] if fingerprint.companies_named else "UNKNOWN"

    direction = "neutral"
    if re.search(r"\b(long|bullish|buy)\b", lower):
        direction = "long"
    elif re.search(r"\b(short|bearish|sell)\b", lower):
        direction = "short"

    strategy_tag = "none"
    strategy_patterns = {
        "momentum": r"\bmomentum\b",
        "mean_reversion": r"\bmean[\s_-]?reversion\b",
        "event_driven": r"\bevent[\s_-]?driven\b",
        "macro": r"\bmacro\b",
    }
    for tag, pattern in strategy_patterns.items():
        if re.search(pattern, lower):
            strategy_tag = tag
            break

    confidence = 0.5
    conf_match = re.search(r"\bconfidence\b[^0-9]*(\d+(?:\.\d+)?)\s*%?", lower)
    if conf_match:
        conf_val = float(conf_match.group(1))
        confidence = conf_val / 100.0 if conf_val > 1.0 else conf_val
    confidence = max(0.0, min(1.0, confidence))

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    cot = " ".join(sentences[:3]).strip()
    if not cot:
        cot = (
            f"Fallback parse used. Direction={direction}, "
            f"strategy_tag={strategy_tag}, confidence={confidence:.2f}."
        )

    return {
        "ticker": ticker,
        "direction": direction,
        "strategy_tag": strategy_tag,
        "confidence": confidence,
        "cot": cot,
    }


def _build_prompt(fingerprint: NewsFingerprint) -> str:
    payload = json.dumps(
        {
            "source": fingerprint.source,
            "published_at": fingerprint.published_at,
            "headline": fingerprint.headline,
            "companies_named": fingerprint.companies_named,
            "event_keywords": fingerprint.event_keywords,
        },
        indent=2,
    )
    sentiment_line = (
        f"Agent 1 assessed this news as {fingerprint.sentiment_label} "
        f"with a sentiment score of {fingerprint.sentiment_score:.2f} "
        f"(confidence: {fingerprint.sentiment_confidence:.2f}). "
        "Based on this and the extracted facts, determine the trading signal."
    )

    user_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{sentiment_line}\n\n"
        "NewsFingerprint JSON:\n"
        f"{payload}"
    )
    # Match eval behavior: one user message, formatted by chat template.
    return _format_chat_prompt("", user_content)


def _log_thinking(thinking_text: str, ticker: str) -> None:
    """Persist reasoning text to logs/cot_<ticker>_<timestamp>.txt."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = os.path.join(LOGS_DIR, f"cot_{ticker}_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(thinking_text)
    logger.info("Reasoning text saved to %s", filename)


def _save_md_debug_output(
    fingerprint: NewsFingerprint,
    raw_output: str,
    attempt_idx: int,
    token_budget: int,
) -> None:
    """
    Persist raw signal-generation output for debugging in markdown files.
    """
    try:
        base_dir = os.getenv(_DIAG_MD_DIR_ENV, "output/diagnostics_md")
        out_dir = os.path.join(base_dir, "agent2")
        os.makedirs(out_dir, exist_ok=True)

        identity = f"{fingerprint.headline}|{fingerprint.companies_named[0] if fingerprint.companies_named else ''}"
        fp_id = md5(identity.encode("utf-8")).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp}_a2_{fp_id}_attempt{attempt_idx}_tok{token_budget}.md"
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(raw_output)
        logger.info("Saved Agent 2 markdown debug output to %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save Agent 2 markdown debug output: %s", exc)


def generate_signal(fingerprint: NewsFingerprint) -> Optional[TradingSignal]:
    """
    Run vLLM inference to generate and validate a TradingSignal.

    Returns None on any model, parse, IO, or validation failure.
    """
    try:
        _load_model()
        from vllm import SamplingParams  # type: ignore

        prompt = _build_prompt(fingerprint)
        signal_schema = TradingSignal.model_json_schema()
        params_kwargs: dict[str, Any] = {
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        try:
            from vllm.sampling_params import StructuredOutputParams  # type: ignore

            params_kwargs["guided_decoding"] = StructuredOutputParams(json=signal_schema)
        except ImportError:
            logger.warning(
                "Structured output params unavailable in current vLLM version; "
                "falling back to non-guided signal generation."
            )

        params = SamplingParams(**params_kwargs)
        outputs = _vllm_engine.generate([prompt], params)
        raw_text = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
        clean_text = _normalize_generated_text(raw_text)
        _save_md_debug_output(fingerprint, clean_text, 1, 256)
        logger.info(
            "Agent 2 raw output (attempt %d, max_tokens=%d): %s",
            1,
            256,
            clean_text[:300],
        )

        parsed = _parse_signal_structured(clean_text)

        signal = TradingSignal(**parsed)

        reasoning_text = parsed.get("cot", "").strip() if isinstance(parsed, dict) else ""
        if reasoning_text:
            _log_thinking(reasoning_text, signal.ticker)
        else:
            fallback = (
                f"direction={signal.direction}; strategy_tag={signal.strategy_tag}; "
                f"confidence={signal.confidence:.2f}; no cot field content returned."
            )
            _log_thinking(fallback, signal.ticker)

        return signal

    except Exception as exc:
        logger.error("generate_signal failed: %s", exc, exc_info=True)
        return None
