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
from typing import Optional

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

_vllm_engine = None


def _load_model() -> None:
    """Lazy-load a vLLM engine once; reuse Agent 1 engine in shared mode."""
    global _vllm_engine  # noqa: PLW0603

    if _vllm_engine is not None:
        return

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
    logger.info("Injected shared vLLM engine into Agent 2 reasoner.")


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _extract_json_blob(text: str) -> str:
    cleaned = _strip_code_fences(text)
    match = _JSON_OBJ_RE.search(cleaned)
    return match.group(0).strip() if match else cleaned


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

    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        "<|user|>\n"
        f"{sentiment_line}\n\n"
        "NewsFingerprint JSON:\n"
        f"{payload}\n"
        "<|assistant|>\n"
    )


def _log_thinking(thinking_text: str, ticker: str) -> None:
    """Persist reasoning text to logs/cot_<ticker>_<timestamp>.txt."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = os.path.join(LOGS_DIR, f"cot_{ticker}_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(thinking_text)
    logger.info("Reasoning text saved to %s", filename)


def generate_signal(fingerprint: NewsFingerprint) -> Optional[TradingSignal]:
    """
    Run vLLM inference to generate and validate a TradingSignal.

    Returns None on any model, parse, IO, or validation failure.
    """
    try:
        _load_model()
        from vllm import SamplingParams  # type: ignore

        prompt = _build_prompt(fingerprint)
        params = SamplingParams(
            max_tokens=768,
            temperature=0.0,
        )
        outputs = _vllm_engine.generate([prompt], params)
        raw_text = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
        logger.info("Agent 2 raw output: %s", raw_text[:300])

        parsed = json.loads(_extract_json_blob(raw_text))
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
