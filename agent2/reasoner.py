"""
agent2/reasoner.py — Agent 2: real-logprobs trading direction classifier.

The LLM performs two-phase inference (CoT mode) or single-phase (no-CoT mode):

  CoT mode (FINGPT_SIGNAL_USE_COT=True):
    Phase 1 — Chain-of-thought reasoning (greedy, temperature=0.0, stop at </think>).
    Phase 2 — Scoring: one prompt_logprobs call per choice token (A / B / C).

  No-CoT mode (FINGPT_SIGNAL_USE_COT=False, default):
    Phase 1 — Skipped.
    Phase 2 — A/B/C logprobs scored directly against the compact prompt.

All decision logic is deterministic Python:
  adjusted_logits = raw_logits − pmi_alpha * null_logprobs  # PMI bias correction
  probs           = softmax(adjusted_logits / CALIBRATION_T)
  choice          = STRATEGY_SET[argmax(probs)]              # "BUY" | "HOLD" | "SELL"
  direction       = _STRATEGY_DIRECTION[choice]              # "long" | "neutral" | "short"
  strategy_tag    = "event_driven"                           # always fixed

Confidence/margin filters are applied after softmax:
  - top_prob < FINGPT_SIGNAL_MIN_CONFIDENCE  → force HOLD
  - margin   < FINGPT_SIGNAL_MIN_MARGIN      → force HOLD
  - top==A and prob[A] < FINGPT_BUY_THRESHOLD  → force HOLD
  - top==C and prob[C] < FINGPT_SELL_THRESHOLD → force HOLD

PMI null logprobs are computed once per model, then persisted to PMI_PRIOR_PATH.
The cache key excludes post-processing parameters (pmi_alpha, calibration_T,
thresholds) so they can be tuned without invalidating the cache.
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
    FINGPT_BUY_THRESHOLD,
    FINGPT_MODEL_PATH,
    FINGPT_PMI_ALPHA,
    FINGPT_SELL_THRESHOLD,
    FINGPT_SIGNAL_MIN_CONFIDENCE,
    FINGPT_SIGNAL_MIN_MARGIN,
    FINGPT_SIGNAL_USE_COT,
    LOG_LEVEL,
    LOGITS_MAX_TOKENS,
    LOGS_DIR,
    PMI_PRIOR_PATH,
    SHARE_SINGLE_LLM_BETWEEN_AGENTS,
    STRATEGY_SET,
)
from agent1.schema import NewsFingerprint
from agent2.prompt import (
    STRATEGY_DECISION_PREFIX,
    STRATEGY_PROMPT_COT,
    STRATEGY_PROMPT_NO_COT,
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

# strategy_tag is fixed for all news-driven signals.
_FIXED_STRATEGY_TAG = "event_driven"

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
    """
    Return the metadata dict used to validate a saved PMI prior file.

    Only model/tokenizer/prompt-structure fields are included.
    Post-processing parameters (pmi_alpha, calibration_T, thresholds) are
    intentionally excluded — they do not affect the null-context logprobs
    and should not invalidate the cache.
    """
    return {
        "model_path": FINGPT_MODEL_PATH,
        "decision_prefix": STRATEGY_DECISION_PREFIX,
        "score_tokens": STRATEGY_SCORE_TOKENS,
    }


def _load_null_logprobs_from_disk() -> Optional[list[float]]:
    """
    Try to load a previously saved PMI prior from ``PMI_PRIOR_PATH``.

    The file is only accepted when its ``model_path``, ``decision_prefix``,
    and ``score_tokens`` fields match the current configuration.
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
      3. Run a vLLM inference call with a null/neutral article.
    """
    if _vllm_engine is None or _chat_tokenizer is None:
        return None

    cached = _load_null_logprobs_from_disk()
    if cached is not None:
        return cached

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
        event_type="MACRO",
        event_type_confidence=0.333,
        event_type_margin=0.0,
        event_type_method="logits_accepted",
        article_text="No specific news available for this period.",
    )
    try:
        null_prompt = _build_prompt(null_fp)
        result = get_real_choice_logits(
            engine=_vllm_engine,
            cot_prompt=null_prompt,
            choices=STRATEGY_SCORE_TOKENS,
            decision_prefix=STRATEGY_DECISION_PREFIX,
            tokenizer=_chat_tokenizer,
            max_cot_tokens=256,
            use_cot=FINGPT_SIGNAL_USE_COT,
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
    Build the Agent 2 direction-classification prompt from the fingerprint.

    Uses the compact no-CoT or CoT variant of STRATEGY_PROMPT depending on
    FINGPT_SIGNAL_USE_COT.  Agent 2 receives only the ticker, headline, and
    structured Agent 1 outputs — it never reads the full article text.
    """
    template = STRATEGY_PROMPT_COT if FINGPT_SIGNAL_USE_COT else STRATEGY_PROMPT_NO_COT

    ticker = fingerprint.companies_named[0] if fingerprint.companies_named else "UNKNOWN"
    sp = fingerprint.sentiment_probabilities or {}
    p_pos = round(sp.get("POSITIVE", 0.0), 4)
    p_neu = round(sp.get("NEUTRAL", 0.0), 4)
    p_neg = round(sp.get("NEGATIVE", 0.0), 4)

    user_content = template.format(
        ticker=ticker,
        headline=fingerprint.headline or "No headline.",
        sentiment_label=fingerprint.sentiment_label,
        sentiment_confidence=round(fingerprint.sentiment_confidence, 4),
        p_pos=p_pos,
        p_neu=p_neu,
        p_neg=p_neg,
        event_type=fingerprint.event_type,
        event_type_confidence=(
            round(fingerprint.event_type_confidence, 4)
            if fingerprint.event_type_confidence is not None else "n/a"
        ),
        event_type_margin=(
            round(fingerprint.event_type_margin, 4)
            if fingerprint.event_type_margin is not None else "n/a"
        ),
        event_type_method=fingerprint.event_type_method or "n/a",
        companies_named=", ".join(fingerprint.companies_named) or "n/a",
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
    """Write a diagnostic file when logits JSON parsing fails."""
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
    Convert a ``get_real_choice_logits`` result dict into a ``TradingSignal``.

    Pipeline:
      1. PMI alpha correction: adjusted = raw - pmi_alpha * null
      2. Temperature softmax: probs = softmax(adjusted / CALIBRATION_T)
      3. Confidence/margin/direction filters (force HOLD when triggered)
      4. Assemble TradingSignal with diagnostics

    strategy_tag is always "event_driven" — it is not predicted by the model.

    Returns ``None`` on parse failure.
    """
    if not result["parse_success"] or result["logits"] is None:
        logger.error(
            "Agent 2 logits parse failed for headline: %s", fingerprint.headline
        )
        _save_failure_diagnostic(fingerprint, result["raw_output"])
        return None

    raw_logits: list[float] = result["logits"]

    # PMI prior correction with configurable alpha.
    # alpha=1.0 → full PMI (original behaviour).
    # alpha=0.0 → no correction (raw logits only).
    if _null_logprobs is not None and len(_null_logprobs) == len(raw_logits):
        pmi_alpha = FINGPT_PMI_ALPHA
        adjusted_logits = [r - pmi_alpha * n for r, n in zip(raw_logits, _null_logprobs)]
        logger.debug(
            "PMI correction applied (alpha=%.2f): raw=%s null=%s adjusted=%s",
            pmi_alpha, raw_logits, _null_logprobs, adjusted_logits,
        )
    else:
        adjusted_logits = raw_logits

    probs = softmax(adjusted_logits, temperature=CALIBRATION_T)
    best_idx = probs.index(max(probs))
    raw_best_label: str = STRATEGY_SCORE_TOKENS[best_idx]  # "A", "B", or "C"
    strategy: str = STRATEGY_SET[best_idx]                  # "BUY", "HOLD", or "SELL"
    top_prob: float = probs[best_idx]
    probabilities = dict(zip(STRATEGY_SET, probs))

    # Compute margin (top − second).
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

    # ── Confidence / margin / direction filters ───────────────────────────────
    forced_hold = False
    filter_reason: Optional[str] = None

    if top_prob < FINGPT_SIGNAL_MIN_CONFIDENCE:
        forced_hold = True
        filter_reason = "low_confidence"
    elif margin < FINGPT_SIGNAL_MIN_MARGIN:
        forced_hold = True
        filter_reason = "low_margin"
    elif raw_best_label == "A" and top_prob < FINGPT_BUY_THRESHOLD:
        forced_hold = True
        filter_reason = "buy_threshold"
    elif raw_best_label == "C" and top_prob < FINGPT_SELL_THRESHOLD:
        forced_hold = True
        filter_reason = "sell_threshold"

    if forced_hold:
        strategy = "HOLD"
        logger.info(
            "Signal filter forced HOLD (reason=%s): raw_top=%s top_prob=%.4f margin=%.4f",
            filter_reason, raw_best_label, top_prob, margin,
        )

    direction: str = _STRATEGY_DIRECTION[strategy]
    confidence: float = probabilities[strategy]

    logger.info(
        "Agent 2 raw_logits=%s  adjusted=%s  probs=%s  raw_top=%s  final=%s  "
        "dir=%s  conf=%.4f  margin=%.4f  filter=%s",
        raw_logits,
        adjusted_logits,
        [f"{p:.4f}" for p in probs],
        raw_best_label,
        strategy,
        direction,
        confidence,
        margin,
        filter_reason or "none",
    )

    thinking_text = result.get("thinking", "").strip()
    ticker = fingerprint.companies_named[0] if fingerprint.companies_named else "UNKNOWN"

    if thinking_text:
        _log_thinking(thinking_text, ticker)

    return TradingSignal(
        ticker=ticker,
        direction=direction,  # type: ignore[arg-type]
        strategy_tag=_FIXED_STRATEGY_TAG,  # type: ignore[arg-type]
        confidence=round(confidence, 6),
        cot=thinking_text,
        signal_logits=adjusted_logits,
        raw_signal_logits=raw_logits,
        signal_probabilities={k: round(v, 6) for k, v in probabilities.items()},
        calibration_T=CALIBRATION_T,
        signal_filter_forced_hold=forced_hold,
        signal_filter_reason=filter_reason,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_signal_batch(
    fingerprints: list[NewsFingerprint],
) -> list[Optional[TradingSignal]]:
    """
    Batch version of ``generate_signal`` using real model logprobs.

    In CoT mode (FINGPT_SIGNAL_USE_COT=True), makes two batched calls:
      Phase 1 — N CoT generations (stop at </think>).
      Phase 2 — N x K prompt_logprobs scoring calls.

    In no-CoT mode (default), makes one batched call:
      Phase 2 only — N x K prompt_logprobs scoring calls.
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

    if _null_logprobs is None:
        _null_logprobs = _compute_null_logprobs()

    cot_prompts = [_build_prompt(fp) for fp in fingerprints]

    results = get_real_choice_logits_batch(
        engine=_vllm_engine,
        cot_prompts=cot_prompts,
        choices=STRATEGY_SCORE_TOKENS,
        decision_prefix=STRATEGY_DECISION_PREFIX,
        tokenizer=_chat_tokenizer,
        max_cot_tokens=LOGITS_MAX_TOKENS,
        use_cot=FINGPT_SIGNAL_USE_COT,
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

    In CoT mode, two vLLM calls are made (Phase 1 CoT + Phase 2 scoring).
    In no-CoT mode (default), one vLLM call is made (Phase 2 only).

    Returns None on any model, parse, or validation failure.
    """
    global _null_logprobs  # noqa: PLW0603
    try:
        _load_model()

        if _chat_tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded — cannot compute real choice logprobs."
            )

        if _null_logprobs is None:
            _null_logprobs = _compute_null_logprobs()

        cot_prompt = _build_prompt(fingerprint)
        result = get_real_choice_logits(
            engine=_vllm_engine,
            cot_prompt=cot_prompt,
            choices=STRATEGY_SCORE_TOKENS,
            decision_prefix=STRATEGY_DECISION_PREFIX,
            tokenizer=_chat_tokenizer,
            max_cot_tokens=LOGITS_MAX_TOKENS,
            use_cot=FINGPT_SIGNAL_USE_COT,
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
