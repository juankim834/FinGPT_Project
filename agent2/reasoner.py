"""
agent2/reasoner.py — Agent 2 local HuggingFace reasoner.

Converts a NewsFingerprint into a TradingSignal using a locally loaded
DeepSeek model. Raw reasoning text is persisted to logs/cot_<ticker>_<timestamp>.txt.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import torch

from config import (
    FINGPT_ADAPTER_PATH,
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

_tokenizer = None
_model = None


def _load_model() -> None:
    """Lazy-load a dedicated Agent 2 local model/tokenizer once."""
    global _tokenizer, _model  # noqa: PLW0603

    if _model is not None and _tokenizer is not None:
        return

    if SHARE_SINGLE_LLM_BETWEEN_AGENTS:
        from agent1.extractor import get_loaded_model_and_tokenizer

        _model, _tokenizer = get_loaded_model_and_tokenizer()
        if _model is not None and _tokenizer is not None:
            logger.info("Agent 2 reusing shared model instance from Agent 1.")
            return
        logger.info(
            "Agent 1 is running on vLLM without HF model/tokenizer export; "
            "Agent 2 will load its dedicated local HF model."
        )

    if not FINGPT_MODEL_PATH:
        raise EnvironmentError(
            "FINGPT_MODEL_PATH is not set. Add it to your .env pointing to local weights."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    has_cuda = torch.cuda.is_available()
    runtime_device = torch.device("cuda" if has_cuda else "cpu")
    torch_dtype = torch.bfloat16 if has_cuda else torch.float32

    logger.info("Loading Agent 2 model from: %s", FINGPT_MODEL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(FINGPT_MODEL_PATH)
    _model = AutoModelForCausalLM.from_pretrained(
        FINGPT_MODEL_PATH,
        torch_dtype=torch_dtype,
        device_map="auto" if has_cuda else None,
    )
    if FINGPT_ADAPTER_PATH:
        from peft import PeftModel  # type: ignore

        logger.info("Applying LoRA adapter from: %s", FINGPT_ADAPTER_PATH)
        _model = PeftModel.from_pretrained(_model, FINGPT_ADAPTER_PATH)
    if not has_cuda:
        _model.to(runtime_device)
    _model.eval()
    logger.info("Agent 2 local model loaded on %s", runtime_device)


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
    Run local HF inference to generate and validate a TradingSignal.

    Returns None on any model, parse, IO, or validation failure.
    """
    try:
        _load_model()
        prompt = _build_prompt(fingerprint)

        model_device = next(_model.parameters()).device
        inputs = _tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                temperature=0.0,
                pad_token_id=_tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = _tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
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
