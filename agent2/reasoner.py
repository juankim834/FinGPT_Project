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

    if _vllm_engine is not None and _chat_tokenizer is not None:
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
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    result: dict[str, Any] = {
        "ticker": "",
        "direction": "neutral",
        "strategy_tag": "none",
        "confidence": 0.0,
        "cot": "",
    }

    cot_started = False
    for line in lines:
        lower = line.lower()
        if lower.startswith("### "):
            continue
        if lower.startswith("- ticker:"):
            result["ticker"] = line.split(":", 1)[1].strip()
            cot_started = False
            continue
        if lower.startswith("- direction:"):
            result["direction"] = line.split(":", 1)[1].strip().lower()
            cot_started = False
            continue
        if lower.startswith("- strategy_tag:"):
            result["strategy_tag"] = line.split(":", 1)[1].strip().lower()
            cot_started = False
            continue
        if lower.startswith("- confidence:"):
            raw_conf = line.split(":", 1)[1].strip()
            try:
                result["confidence"] = float(raw_conf)
            except ValueError:
                # Keep default and let schema validation catch if needed.
                pass
            cot_started = False
            continue
        if lower.startswith("- cot:"):
            result["cot"] = line.split(":", 1)[1].strip()
            cot_started = True
            continue
        if cot_started and not lower.startswith("- "):
            # Allow wrapped reasoning lines after `- cot:`.
            result["cot"] = f"{result['cot']} {line}".strip()

    return result


def _parse_signal_structured(raw_text: str) -> dict[str, Any]:
    try:
        return _parse_json_signal(raw_text)
    except Exception:
        parsed = _parse_markdown_signal(raw_text)
        if parsed.get("ticker") and parsed.get("cot"):
            return parsed
        raise


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
        f"{sentiment_line}\n\n"
        "NewsFingerprint JSON:\n"
        f"{payload}\n"
    )
    return _format_chat_prompt(SYSTEM_PROMPT, user_content)


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
        token_budgets = [768, 1024]
        parsed: dict[str, Any] | None = None
        last_parse_error: Exception | None = None

        for attempt_idx, token_budget in enumerate(token_budgets, start=1):
            params = SamplingParams(
                max_tokens=token_budget,
                temperature=0.0,
            )
            outputs = _vllm_engine.generate([prompt], params)
            raw_text = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
            _save_md_debug_output(fingerprint, raw_text, attempt_idx, token_budget)
            logger.info(
                "Agent 2 raw output (attempt %d, max_tokens=%d): %s",
                attempt_idx,
                token_budget,
                raw_text[:300],
            )
            try:
                parsed = _parse_signal_structured(raw_text)
                break
            except Exception as exc:
                last_parse_error = exc
                logger.warning(
                    "Agent 2 parse failed (attempt %d, max_tokens=%d): %s",
                    attempt_idx,
                    token_budget,
                    exc,
                )

        if parsed is None:
            raise RuntimeError("Agent 2 failed to produce valid structured output.") from last_parse_error

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
