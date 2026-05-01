"""
agent2/reasoner.py — Agent 2: real-logprobs trading strategy reasoner.

The LLM performs two-phase inference:
  Phase 1 — Chain-of-thought reasoning (greedy, temperature=0.0, stop at </think>).
  Phase 2 — Scoring: one prompt_logprobs call per choice token (A / B / C).

All decision logic is deterministic Python:
  pmi_logits = raw_logits − null_logprobs          # PMI bias correction
  probs      = softmax(pmi_logits / CALIBRATION_T)
  choice     = STRATEGY_SET[argmax(probs)]         # "BUY" | "HOLD" | "SELL"
  direction  = _STRATEGY_DIRECTION[choice]          # "long" | "neutral" | "short"
  confidence = max(probs)

PMI null logprobs (the model's unconditional prior at "Strategy: ") are
computed once per model, then persisted to PMI_PRIOR_PATH (JSON) so that
subsequent sessions skip the extra vLLM call.

generate_signal() returns None on any inference or validation failure.
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
    FINGPT_MODEL_PATH,
    LOG_LEVEL,
    LOGITS_MAX_TOKENS,
    LOGS_DIR,
    PMI_PRIOR_PATH,
    SHARE_SINGLE_LLM_BETWEEN_AGENTS,
    STRATEGY_SET,
)
from agent1.schema import NewsFingerprint
from agent2.prompt import (
    STRATEGY_COT_PROMPT,
    STRATEGY_DECISION_PREFIX,
    STRATEGY_SCORE_TOKENS,
)
from agent2.schema import TradingSignal
from vllm_logits_client import (
    get_real_choice_logits,
    get_real_choice_logits_batch,
    softmax,
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

_DIAG_MD_DIR_ENV = "FINGPT_DIAG_MD_DIR"
_SPECIAL_TOKEN_RE = re.compile(
    r"(<\|[^>]+\|>|<｜[^｜]+｜>|<s>|</s>|\[INST\]|\[/INST\])"
)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

# Maps the selected strategy to the TradingSignal direction.
_STRATEGY_DIRECTION: dict[str, str] = {
    "BUY": "long",
    "HOLD": "neutral",
    "SELL": "short",
}

# Maps the selected strategy to a strategy_tag value.
_STRATEGY_TAG: dict[str, str] = {
    "BUY": "event_driven",
    "HOLD": "none",
    "SELL": "event_driven",
}

_vllm_engine = None
_chat_tokenizer = None
_null_logprobs: Optional[list[float]] = None   # PMI prior — computed once per engine load


# ---------------------------------------------------------------------------
# Engine management
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
    """Lazy-load a vLLM engine; reuse Agent 1 engine in shared mode."""
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
    global _vllm_engine, _null_logprobs  # noqa: PLW0603
    _vllm_engine = engine
    _null_logprobs = None   # reset so null logprobs are re-computed for the new engine
    _ensure_chat_tokenizer()
    logger.info("Injected shared vLLM engine into Agent 2 reasoner.")


# ---------------------------------------------------------------------------
# PMI prior correction
# ---------------------------------------------------------------------------

def _pmi_cache_key() -> dict:
    """Return the metadata dict used to validate a saved PMI prior file."""
    return {
        "model_path": FINGPT_MODEL_PATH,
        "decision_prefix": STRATEGY_DECISION_PREFIX,
        "score_tokens": STRATEGY_SCORE_TOKENS,
    }


def _load_null_logprobs_from_disk() -> Optional[list[float]]:
    """
    Try to load a previously saved PMI prior from ``PMI_PRIOR_PATH``.

    The file is only accepted when its ``model_path``, ``decision_prefix``,
    and ``score_tokens`` fields match the current configuration — guarding
    against accidental reuse after a model or prompt change.

    Returns ``None`` when the file does not exist, is invalid, or the key
    does not match.
    """
    if not PMI_PRIOR_PATH:
        return None
    try:
        with open(PMI_PRIOR_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        key = _pmi_cache_key()
        if (
            data.get("model_path") == key["model_path"]
            and data.get("decision_prefix") == key["decision_prefix"]
            and data.get("score_tokens") == key["score_tokens"]
        ):
            null_lp: list[float] = data["null_logprobs"]
            logger.info(
                "PMI prior loaded from disk (%s): %s", PMI_PRIOR_PATH, null_lp
            )
            return null_lp
        logger.info(
            "PMI prior cache key mismatch — will recompute. "
            "Saved: model=%r prefix=%r tokens=%r  "
            "Current: model=%r prefix=%r tokens=%r",
            data.get("model_path"), data.get("decision_prefix"), data.get("score_tokens"),
            key["model_path"], key["decision_prefix"], key["score_tokens"],
        )
    except FileNotFoundError:
        logger.info("No PMI prior cache found at %s — will compute.", PMI_PRIOR_PATH)
    except Exception as exc:
        logger.warning("Could not read PMI prior cache: %s — will recompute.", exc)
    return None


def _save_null_logprobs_to_disk(null_lp: list[float]) -> None:
    """Persist the PMI prior to ``PMI_PRIOR_PATH`` with a validation key."""
    if not PMI_PRIOR_PATH:
        return
    try:
        os.makedirs(os.path.dirname(PMI_PRIOR_PATH) or ".", exist_ok=True)
        payload = {
            **_pmi_cache_key(),
            "null_logprobs": null_lp,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(PMI_PRIOR_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("PMI prior saved to %s: %s", PMI_PRIOR_PATH, null_lp)
    except Exception as exc:
        logger.warning("Could not save PMI prior to disk: %s", exc)


def _compute_null_logprobs() -> Optional[list[float]]:
    """
    Return the language-model prior log-probabilities for the scoring tokens
    (A / B / C) using a fully neutral article (PMI baseline).

    Load order:
      1. Return the in-memory ``_null_logprobs`` if already set (caller's job).
      2. Try to load a matching prior from ``PMI_PRIOR_PATH`` on disk.
      3. Run a vLLM inference call with a null/neutral article and save the
         result to disk for future sessions.

    Without this correction, one letter consistently dominates because the
    language model assigns an unconditional prior to certain tokens at the
    ``"Strategy: "`` decision position regardless of news content.
    """
    if _vllm_engine is None or _chat_tokenizer is None:
        return None

    # Step 2: try disk cache
    cached = _load_null_logprobs_from_disk()
    if cached is not None:
        return cached

    # Step 3: compute via vLLM and persist
    null_fp = NewsFingerprint(
        source="null",
        published_at="",
        headline="No specific news.",
        companies_named=["UNKNOWN"],
        event_keywords=[],
        sentiment_label="NEUTRAL",
        sentiment_score=0.0,
        sentiment_confidence=0.333,
        sentiment_probabilities={"POSITIVE": 0.333, "NEGATIVE": 0.333, "NEUTRAL": 0.333},
        article_text="No specific news available for this period.",
    )
    try:
        null_cot_prompt = _build_prompt(null_fp)
        result = get_real_choice_logits(
            engine=_vllm_engine,
            cot_prompt=null_cot_prompt,
            choices=STRATEGY_SCORE_TOKENS,
            decision_prefix=STRATEGY_DECISION_PREFIX,
            tokenizer=_chat_tokenizer,
            max_cot_tokens=256,     # short budget — null reasoning only
        )
        null_lp = result["logits"]
        logger.info("PMI null logprobs computed (A/B/C): %s", null_lp)
        _save_null_logprobs_to_disk(null_lp)
        return null_lp
    except Exception as exc:
        logger.warning("Could not compute PMI null logprobs: %s — skipping correction.", exc)
        return None


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_chat_prompt(system_prompt: str, user_content: str) -> str:
    """Format using tokenizer chat template, with safe fallback."""
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


def _build_prompt(fingerprint: NewsFingerprint) -> str:
    """
    Build the Agent 2 strategy-selection prompt.

    The prompt includes:
    - Agent 1's sentiment JSON (label, confidence, full probability vector).
    - The original article text (stored on the fingerprint).
    - The CoT + logits instructions from STRATEGY_COT_PROMPT.

    Falls back to the fingerprint headline when article_text is empty (e.g.
    if an older fingerprint was passed that predates the article_text field).
    """
    sentiment_payload = {
        "sentiment": fingerprint.sentiment_label,
        "confidence": round(fingerprint.sentiment_confidence, 4),
        "probabilities": fingerprint.sentiment_probabilities,
    }
    sentiment_json = json.dumps(sentiment_payload, indent=2)

    article_text = fingerprint.article_text or fingerprint.headline
    if not article_text:
        article_text = (
            f"Headline: {fingerprint.headline}\n"
            f"Companies: {', '.join(fingerprint.companies_named)}\n"
            f"Keywords: {', '.join(fingerprint.event_keywords)}"
        )

    user_content = STRATEGY_COT_PROMPT.format(
        sentiment_json=sentiment_json,
        article_text=article_text,
    )
    return _format_chat_prompt("", user_content)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_thinking(thinking_text: str, ticker: str) -> None:
    """Persist chain-of-thought reasoning to logs/cot_<ticker>_<timestamp>.txt."""
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
    """Persist raw signal-generation output for debugging."""
    try:
        base_dir = os.getenv(_DIAG_MD_DIR_ENV, "output/diagnostics_md")
        out_dir = os.path.join(base_dir, "agent2")
        os.makedirs(out_dir, exist_ok=True)

        identity = (
            f"{fingerprint.headline}|"
            f"{fingerprint.companies_named[0] if fingerprint.companies_named else ''}"
        )
        fp_id = md5(identity.encode("utf-8")).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = (
            f"{timestamp}_a2_{fp_id}_attempt{attempt_idx}_tok{token_budget}.md"
        )
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(raw_output)
        logger.info("Saved Agent 2 debug output to %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save Agent 2 debug output: %s", exc)


def _save_failure_diagnostic(fingerprint: NewsFingerprint, raw_output: str) -> None:
    """Write a diagnostic file when logits JSON parsing fails (strict mode)."""
    try:
        base_dir = os.getenv(_DIAG_MD_DIR_ENV, "output/diagnostics_md")
        out_dir = os.path.join(base_dir, "agent2_failures")
        os.makedirs(out_dir, exist_ok=True)

        identity = (
            f"{fingerprint.headline}|"
            f"{fingerprint.companies_named[0] if fingerprint.companies_named else 'UNKNOWN'}"
        )
        fp_id = md5(identity.encode("utf-8")).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp}_a2_FAILURE_{fp_id}.md"
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(
                f"# Agent 2 logits parse failure\n\n"
                f"**Headline:** {fingerprint.headline}\n\n"
                f"**Companies:** {fingerprint.companies_named}\n\n"
                f"**Expected choices:** {STRATEGY_SET}\n\n"
                f"## Raw model output\n\n```\n{raw_output}\n```\n"
            )
        logger.warning("Failure diagnostic written to %s", out_path)
    except Exception as exc:
        logger.warning("Could not write failure diagnostic: %s", exc)


# ---------------------------------------------------------------------------
# Signal post-processing (shared by single-item and batch paths)
# ---------------------------------------------------------------------------

def _process_signal_result(
    fingerprint: NewsFingerprint,
    result: dict,
) -> Optional[TradingSignal]:
    """
    Convert a ``get_choice_logits`` / ``get_choice_logits_batch`` result dict
    into a ``TradingSignal``.

    This function contains **all** output logic: softmax, argmax, direction
    mapping, strategy tag, confidence, CoT extraction, and the diagnostic
    failure path.  It is called identically by ``generate_signal`` (single-item)
    and ``generate_signal_batch``.

    Returns ``None`` on parse failure (strict mode) or validation error.
    """
    if not result["parse_success"] or result["logits"] is None:
        logger.error(
            "Agent 2 logits parse failed for headline: %s", fingerprint.headline
        )
        _save_failure_diagnostic(fingerprint, result["raw_output"])
        return None

    raw_logits: list[float] = result["logits"]

    # PMI prior correction: subtract null-context logprobs so that the
    # decision is driven by news content rather than a token's unconditional
    # prior at the "Strategy: " decision position.
    # PMI(choice) = logP(choice | article) − logP(choice | null_context)
    if _null_logprobs is not None and len(_null_logprobs) == len(raw_logits):
        logits = [a - b for a, b in zip(raw_logits, _null_logprobs)]
        logger.debug("PMI correction applied: raw=%s null=%s pmi=%s", raw_logits, _null_logprobs, logits)
    else:
        logits = raw_logits

    probs = softmax(logits, temperature=CALIBRATION_T)
    best_idx = probs.index(max(probs))
    strategy: str = STRATEGY_SET[best_idx]
    confidence: float = probs[best_idx]
    probabilities = dict(zip(STRATEGY_SET, probs))

    direction: str = _STRATEGY_DIRECTION[strategy]
    strategy_tag: str = _STRATEGY_TAG[strategy]

    logger.info(
        "Agent 2 raw_logits=%s  pmi_logits=%s  probs=%s  strategy=%s  dir=%s  conf=%.4f",
        raw_logits,
        logits,
        [f"{p:.4f}" for p in probs],
        strategy,
        direction,
        confidence,
    )

    thinking_text = result.get("thinking", "").strip()
    ticker = fingerprint.companies_named[0] if fingerprint.companies_named else "UNKNOWN"

    if thinking_text:
        _log_thinking(thinking_text, ticker)
    else:
        _log_thinking(
            f"strategy={strategy}; direction={direction}; "
            f"confidence={confidence:.4f}; logits={logits}",
            ticker,
        )

    return TradingSignal(
        ticker=ticker,
        direction=direction,  # type: ignore[arg-type]
        strategy_tag=strategy_tag,  # type: ignore[arg-type]
        confidence=round(confidence, 6),
        cot=thinking_text or f"Strategy: {strategy}. Logits: {logits}.",
        signal_logits=logits,
        signal_probabilities={k: round(v, 6) for k, v in probabilities.items()},
        calibration_T=CALIBRATION_T,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_signal_batch(
    fingerprints: list[NewsFingerprint],
) -> list[Optional[TradingSignal]]:
    """
    Batch version of ``generate_signal`` using real model logprobs.

    Makes **two** batched ``engine.generate()`` calls (handled inside
    ``get_real_choice_logits_batch``):
      Phase 1 — N CoT generations (stop at </think>).
      Phase 2 — N × K prompt_logprobs scoring calls.

    Per-item output logic (softmax, argmax, TradingSignal assembly) is
    unchanged and applied via ``_process_signal_result``.
    """
    global _null_logprobs  # noqa: PLW0603
    _load_model()

    if not fingerprints:
        return []

    if _chat_tokenizer is None:
        raise RuntimeError(
            "Tokenizer not loaded — cannot compute real choice logprobs. "
            "Ensure FINGPT_MODEL_PATH is set before calling generate_signal_batch."
        )

    # Compute PMI null logprobs once per engine session (first batch call).
    if _null_logprobs is None:
        _null_logprobs = _compute_null_logprobs()

    cot_prompts = [_build_prompt(fp) for fp in fingerprints]

    results = get_real_choice_logits_batch(
        engine=_vllm_engine,
        cot_prompts=cot_prompts,
        choices=STRATEGY_SCORE_TOKENS,      # ["A","B","C"] → A=BUY, B=HOLD, C=SELL
        decision_prefix=STRATEGY_DECISION_PREFIX,
        tokenizer=_chat_tokenizer,
        max_cot_tokens=LOGITS_MAX_TOKENS,
    )

    signals: list[Optional[TradingSignal]] = []
    for i, (fingerprint, result) in enumerate(zip(fingerprints, results)):
        _save_md_debug_output(fingerprint, result["raw_output"], i + 1, LOGITS_MAX_TOKENS)
        logger.info(
            "Batch signal[%d] real logprobs %s → choices %s",
            i, result["logits"], STRATEGY_SET,
        )
        try:
            signals.append(_process_signal_result(fingerprint, result))
        except Exception as exc:
            logger.error(
                "generate_signal_batch[%d] failed: %s", i, exc, exc_info=True
            )
            signals.append(None)

    return signals


def generate_signal(fingerprint: NewsFingerprint) -> Optional[TradingSignal]:
    """
    Produce a TradingSignal using real model logprobs (single-item path).

    Two vLLM calls are made (Phase 1 CoT + Phase 2 scoring), handled inside
    ``get_real_choice_logits``.  Output logic is delegated to
    ``_process_signal_result`` — unchanged from the previous version.

    Returns None on any model, parse, or validation failure (strict mode).
    """
    global _null_logprobs  # noqa: PLW0603
    try:
        _load_model()

        if _chat_tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded — cannot compute real choice logprobs."
            )

        # Compute PMI null logprobs once per engine session.
        if _null_logprobs is None:
            _null_logprobs = _compute_null_logprobs()

        cot_prompt = _build_prompt(fingerprint)
        result = get_real_choice_logits(
            engine=_vllm_engine,
            cot_prompt=cot_prompt,
            choices=STRATEGY_SCORE_TOKENS,  # ["A","B","C"] → A=BUY, B=HOLD, C=SELL
            decision_prefix=STRATEGY_DECISION_PREFIX,
            tokenizer=_chat_tokenizer,
            max_cot_tokens=LOGITS_MAX_TOKENS,
        )

        _save_md_debug_output(fingerprint, result["raw_output"], 1, LOGITS_MAX_TOKENS)
        logger.info(
            "Agent 2 real logprobs %s → choices %s",
            result["logits"], STRATEGY_SET,
        )

        return _process_signal_result(fingerprint, result)

    except Exception as exc:
        logger.error("generate_signal failed: %s", exc, exc_info=True)
        return None
