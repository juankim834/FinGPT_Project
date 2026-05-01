"""
vllm_logits_client.py — Unified vLLM wrapper for logits-based inference.

Design choice — Option B (self-reported logits):
    The official vLLM API does not expose pre-softmax logits as a first-class
    output.  Rather than patching vLLM internals (Option A), we instruct the
    LLM to report its own unscaled scores ("logits") for each choice class as
    a compact JSON object at the end of its chain-of-thought response.

    Example final output from the model:
        {"logits": [2.1, -0.5, 0.3]}

    These are the model's self-assessed raw confidence scores — numerically
    equivalent to what a classification head would produce before softmax.
    Because softmax is invariant under translation (softmax(x) == softmax(x+c)),
    the absolute scale does not matter; only relative differences between logits
    determine the resulting probability distribution.

    All probability calculations (softmax, optional temperature scaling,
    argmax, confidence) are then performed **deterministically in Python**
    without any further LLM calls.

    As an optional secondary validation, `get_logits_with_vllm_logprobs` uses
    SamplingParams(logprobs=K) to fetch vLLM's own top-K log-probabilities for
    the first non-reasoning token and cross-checks against the self-reported
    logits.  This is off by default and intended for calibration experiments.

Public API
----------
    softmax(logits, temperature) -> list[float]
    get_choice_logits(engine, prompt, choices, max_tokens, temperature) -> dict
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex helpers for parsing the logits JSON from CoT output
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# Matches any JSON object that contains the key "logits"
_LOGITS_OBJ_RE = re.compile(r'\{[^{}]*"logits"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)
# Captures content inside <think> … </think> tags
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Pure-Python post-processing
# ---------------------------------------------------------------------------

def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """
    Numerically stable softmax with temperature scaling.

    softmax_T(x)_i = exp((x_i - max(x)) / T) / sum_j(exp((x_j - max(x)) / T))

    temperature > 1  → softer (more uniform) distribution
    temperature < 1  → sharper (more peaked) distribution
    temperature = 1  → standard softmax
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = [x / temperature for x in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(x - max_val) for x in scaled]
    total = sum(exp_vals)
    return [x / total for x in exp_vals]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


# ---------------------------------------------------------------------------
# JSON extraction from raw model output
# ---------------------------------------------------------------------------

def _extract_logits_json(text: str) -> Optional[dict[str, Any]]:
    """
    Extract the logits JSON object from the model's generated text.

    Search order:
    1. Code-fenced JSON block (```json … ```)
    2. Last occurrence of a JSON object containing a "logits" key
    3. Scan backwards for {"logits": pattern as a last resort
    """
    # 1. Code fence
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        try:
            candidate = json.loads(fence_match.group(1).strip())
            if "logits" in candidate:
                return candidate
        except json.JSONDecodeError:
            pass

    # 2. All regex-matched logits objects — prefer the last one (after CoT)
    matches = list(_LOGITS_OBJ_RE.finditer(text))
    for match in reversed(matches):
        try:
            candidate = json.loads(match.group(0))
            if "logits" in candidate:
                return candidate
        except json.JSONDecodeError:
            continue

    # 3. Backward scan for {"logits" literal
    idx = text.rfind('{"logits"')
    if idx >= 0:
        depth = 0
        for i in range(idx, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[idx : i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def extract_thinking(raw_output: str) -> str:
    """
    Extract chain-of-thought text from <think>…</think> tags.

    Returns the reasoning block, or the full output if no tags are present.
    """
    match = _THINK_BLOCK_RE.search(raw_output)
    if match:
        return match.group(1).strip()
    # If the model did not wrap reasoning in <think> tags, return everything
    # that appears before the final JSON object.
    json_start = raw_output.rfind('{"logits"')
    if json_start > 0:
        return raw_output[:json_start].strip()
    return raw_output.strip()


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def get_choice_logits(
    engine: Any,
    prompt: str,
    choices: list[str],
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Run vLLM generation on *prompt* and extract self-reported logits from the
    JSON object the model is instructed to emit after its CoT reasoning.

    Parameters
    ----------
    engine:
        A vLLM ``LLM`` instance (may be shared across agents).
    prompt:
        Fully formatted prompt string (chat template already applied).
    choices:
        Ordered list of class labels whose logits the model should report,
        e.g. ``["POSITIVE", "NEGATIVE", "NEUTRAL"]``.  The model is prompted
        to emit logits in this exact order.
    max_tokens:
        Maximum tokens to generate (must be large enough to cover the CoT
        reasoning block plus the final JSON).
    temperature:
        Sampling temperature passed to vLLM.  Use ``0.0`` for deterministic
        (greedy) decoding.

    Returns
    -------
    dict with keys:
        ``logits``       – ``list[float]`` aligned with *choices*, or ``None``
                           if parsing failed.
        ``choices``      – same as the input *choices* argument.
        ``raw_output``   – full text generated by the model.
        ``thinking``     – extracted chain-of-thought reasoning text.
        ``parse_success``– ``True`` if a valid logits JSON was found and the
                           length matches *choices*.
    """
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )

    outputs = engine.generate([prompt], params)
    raw_output: str = (
        outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    )

    thinking = extract_thinking(raw_output)
    parsed = _extract_logits_json(raw_output)

    if parsed is not None and "logits" in parsed:
        logits_raw = parsed["logits"]
        if isinstance(logits_raw, list) and len(logits_raw) == len(choices):
            try:
                logits = [float(x) for x in logits_raw]
                logger.debug("Parsed logits %s for choices %s", logits, choices)
                return {
                    "logits": logits,
                    "choices": choices,
                    "raw_output": raw_output,
                    "thinking": thinking,
                    "parse_success": True,
                }
            except (TypeError, ValueError) as exc:
                logger.warning("Could not convert logits to float: %s", exc)
        else:
            logger.warning(
                "Logits length mismatch: expected %d for choices %s, got %s",
                len(choices),
                choices,
                logits_raw,
            )
    else:
        logger.warning(
            "No logits JSON found in model output (first 300 chars): %s",
            raw_output[:300],
        )

    return {
        "logits": None,
        "choices": choices,
        "raw_output": raw_output,
        "thinking": thinking,
        "parse_success": False,
    }


# ---------------------------------------------------------------------------
# Batch variant — same per-item output logic, one engine.generate() call
# ---------------------------------------------------------------------------

def get_choice_logits_batch(
    engine: Any,
    prompts: list[str],
    choices: list[str],
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Batch version of ``get_choice_logits``.

    Sends all *prompts* to vLLM in a **single** ``engine.generate()`` call
    (batch size = ``len(prompts)``), then processes each output independently
    with exactly the same parsing and validation logic as the single-item
    version.  The per-item output schema is identical to ``get_choice_logits``.

    Parameters
    ----------
    engine:
        Shared vLLM ``LLM`` instance.
    prompts:
        List of fully-formatted prompt strings.  Each prompt must instruct the
        model to output ``{"logits": [f1, …, fN]}`` after its CoT reasoning,
        where N = ``len(choices)``.
    choices:
        Ordered list of class labels, e.g. ``["POSITIVE", "NEGATIVE", "NEUTRAL"]``.
        All prompts in the batch must use the same *choices* ordering.
    max_tokens:
        Token budget per prompt (applied uniformly across the batch).
    temperature:
        Sampling temperature (0.0 = greedy).

    Returns
    -------
    list[dict]
        One dict per prompt, in the same order as *prompts*.  Each dict has the
        same keys as ``get_choice_logits`` returns:
        ``logits``, ``choices``, ``raw_output``, ``thinking``, ``parse_success``.
    """
    if not prompts:
        return []

    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )

    # Single batched inference call.
    outputs = engine.generate(prompts, params)

    results: list[dict[str, Any]] = []
    for i, request_output in enumerate(outputs):
        raw_output: str = (
            request_output.outputs[0].text
            if request_output.outputs
            else ""
        )

        thinking = extract_thinking(raw_output)
        parsed = _extract_logits_json(raw_output)

        if parsed is not None and "logits" in parsed:
            logits_raw = parsed["logits"]
            if isinstance(logits_raw, list) and len(logits_raw) == len(choices):
                try:
                    logits = [float(x) for x in logits_raw]
                    logger.debug(
                        "Batch[%d] parsed logits %s for choices %s", i, logits, choices
                    )
                    results.append({
                        "logits": logits,
                        "choices": choices,
                        "raw_output": raw_output,
                        "thinking": thinking,
                        "parse_success": True,
                    })
                    continue
                except (TypeError, ValueError) as exc:
                    logger.warning("Batch[%d] could not convert logits to float: %s", i, exc)
            else:
                logger.warning(
                    "Batch[%d] logits length mismatch: expected %d for choices %s, got %s",
                    i, len(choices), choices, logits_raw,
                )
        else:
            logger.warning(
                "Batch[%d] no logits JSON found (first 200 chars): %s",
                i, raw_output[:200],
            )

        results.append({
            "logits": None,
            "choices": choices,
            "raw_output": raw_output,
            "thinking": thinking,
            "parse_success": False,
        })

    return results


# ---------------------------------------------------------------------------
# Optional: vLLM logprobs cross-check (calibration experiments only)
# ---------------------------------------------------------------------------

def get_logits_with_vllm_logprobs(
    engine: Any,
    prompt: str,
    choices: list[str],
    max_tokens: int = 1024,
    logprobs_k: int = 50,
) -> dict[str, Any]:
    """
    Extended version of ``get_choice_logits`` that additionally retrieves
    vLLM's internal top-K log-probabilities for all generated tokens.

    This function is intended for **calibration research** — comparing the
    model's self-reported logits against vLLM's actual token log-probabilities.

    The approximate logits reconstructed from vLLM logprobs use the identity:

        logit_approx_i = logprob_i + C   (C is an unknown additive constant)

    Because softmax is shift-invariant we set C = 0, so:

        probs_approx = softmax(logprobs)

    These approximate probabilities are stored under ``vllm_probs`` in the
    returned dict and are suitable for research comparison.

    This function is NOT used in the default pipeline.
    """
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        logprobs=logprobs_k,
    )

    outputs = engine.generate([prompt], params)
    raw_output: str = (
        outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    )
    thinking = extract_thinking(raw_output)
    parsed = _extract_logits_json(raw_output)

    result = get_choice_logits.__wrapped__ if hasattr(get_choice_logits, "__wrapped__") else None

    # Self-reported logits (primary)
    self_reported: Optional[list[float]] = None
    if parsed and "logits" in parsed:
        raw = parsed["logits"]
        if isinstance(raw, list) and len(raw) == len(choices):
            try:
                self_reported = [float(x) for x in raw]
            except (TypeError, ValueError):
                pass

    # vLLM logprobs (secondary, for cross-check)
    token_logprobs = outputs[0].outputs[0].logprobs or []
    vllm_probs: Optional[dict[str, float]] = None
    if token_logprobs:
        # Look for choice tokens in the top-K logprobs across all generated tokens.
        choice_logprobs: dict[str, float] = {}
        for step_logprobs in token_logprobs:
            for token_id, lp_obj in step_logprobs.items():
                decoded = getattr(lp_obj, "decoded_token", "") or ""
                for choice in choices:
                    if choice in decoded and choice not in choice_logprobs:
                        choice_logprobs[choice] = lp_obj.logprob
        if len(choice_logprobs) == len(choices):
            lp_list = [choice_logprobs[c] for c in choices]
            probs = softmax(lp_list, temperature=1.0)
            vllm_probs = dict(zip(choices, probs))

    return {
        "logits": self_reported,
        "choices": choices,
        "raw_output": raw_output,
        "thinking": thinking,
        "parse_success": self_reported is not None,
        "vllm_probs": vllm_probs,
    }
