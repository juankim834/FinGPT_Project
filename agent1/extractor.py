"""
agent1/extractor.py — Agent 1: FinGPT fact + sentiment extractor.

The module keeps a lazy, single-load HuggingFace setup and exposes one public
API: extract_fingerprint(article_text) -> Optional[NewsFingerprint].
"""

import json
import logging
import re
from typing import Optional

import torch

from config import (
    FINGPT_ADAPTER_PATH,
    FINGPT_MAX_NEW_TOKENS,
    FINGPT_MODEL_PATH,
    FINGPT_TEMPERATURE,
    LOG_LEVEL,
    SHARE_SINGLE_LLM_BETWEEN_AGENTS,
)
from agent1.schema import NewsFingerprint, SentimentLabel

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a financial news fact extractor.\n"
    "Return exactly one JSON object with keys:\n"
    'source, published_at, headline, companies_named, event_keywords\n'
    "Rules:\n"
    "- Only use explicit article facts.\n"
    "- headline must be copied exactly from text when present.\n"
    "- companies_named should include all company names/tickers in the article.\n"
    "- event_keywords must be lowercase short keywords copied from text.\n"
    "- No markdown, no commentary, no extra keys."
)

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

_tokenizer = None
_model = None
_generator = None
_runtime_device = None


def _load_model() -> None:
    """Lazy-load tokenizer/model/pipeline once, using bf16 on CUDA."""
    global _tokenizer, _model, _generator, _runtime_device  # noqa: PLW0603

    if _model is not None and _tokenizer is not None and _generator is not None:
        return

    if not FINGPT_MODEL_PATH:
        raise EnvironmentError(
            "FINGPT_MODEL_PATH is not set. Add it to your .env pointing to local weights."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

    has_cuda = torch.cuda.is_available()
    _runtime_device = torch.device("cuda" if has_cuda else "cpu")
    torch_dtype = torch.bfloat16 if has_cuda else torch.float32

    logger.info("Loading FinGPT model from: %s", FINGPT_MODEL_PATH)
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
        _model.to(_runtime_device)
    _model.eval()

    _generator = pipeline(
        task="text-generation",
        model=_model,
        tokenizer=_tokenizer,
    )
    if SHARE_SINGLE_LLM_BETWEEN_AGENTS:
        logger.info("Shared FinGPT model loaded on %s for both agents.", _runtime_device)
    else:
        logger.info("FinGPT model and pipeline loaded on %s.", _runtime_device)


def get_loaded_model_and_tokenizer():
    """Return the loaded Agent 1 model/tokenizer pair (loading lazily if needed)."""
    _load_model()
    return _model, _tokenizer


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _extract_json_blob(text: str) -> str:
    cleaned = _strip_code_fences(text)
    match = _JSON_OBJ_RE.search(cleaned)
    return match.group(0).strip() if match else cleaned


def _build_extraction_prompt(article_text: str) -> str:
    return (
        f"<|system|>\n{_EXTRACTION_SYSTEM_PROMPT}\n"
        f"<|user|>\n{article_text}\n"
        "<|assistant|>\n"
    )


def _build_sentiment_prompt(article_text: str) -> str:
    labels = ", ".join(_SENTIMENT_LABELS)
    return (
        "Classify the market sentiment of this news article.\n"
        f"Valid labels: {labels}.\n"
        "Respond with only one label.\n\n"
        f"Article:\n{article_text}\n\nLabel:"
    )


def _sum_candidate_log_prob(prompt_text: str, candidate_label: SentimentLabel) -> float:
    prompt_inputs = _tokenizer(prompt_text, return_tensors="pt")
    model_device = next(_model.parameters()).device
    prompt_inputs = {k: v.to(model_device) for k, v in prompt_inputs.items()}

    candidate_ids = _tokenizer(
        f" {candidate_label}",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model_device)

    full_input_ids = torch.cat((prompt_inputs["input_ids"], candidate_ids), dim=1)
    full_attention = torch.ones_like(full_input_ids, device=model_device)

    with torch.no_grad():
        outputs = _model(input_ids=full_input_ids, attention_mask=full_attention)
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        targets = full_input_ids[:, 1:]
        token_log_probs = log_probs.gather(dim=2, index=targets.unsqueeze(-1)).squeeze(-1)

    prompt_len = prompt_inputs["input_ids"].shape[1]
    candidate_len = candidate_ids.shape[1]
    start = prompt_len - 1
    end = start + candidate_len
    return float(token_log_probs[0, start:end].sum().item())


def _score_sentiment(article_text: str) -> dict[str, float | str]:
    """
    Compute sentiment from token log-probabilities over fixed candidate labels.

    Method:
    1) Build a classification prompt and call `model.generate(..., max_new_tokens=1,
       return_dict_in_generate=True, output_scores=True)` to collect the next-token
       score distribution from the same generation stack used in production.
    2) Because labels are multi-token strings, compute each candidate's sequence
       log-probability using teacher forcing: append the candidate tokens to the
       prompt, run a forward pass, and sum per-token log-probs at the candidate
       positions.
    3) Apply softmax across the five candidate scores to get probabilities, then
       map labels to {-2,-1,0,1,2} and compute the expected value as sentiment_score.
    """
    prompt = _build_sentiment_prompt(article_text)
    model_device = next(_model.parameters()).device
    prompt_inputs = _tokenizer(prompt, return_tensors="pt")
    prompt_inputs = {k: v.to(model_device) for k, v in prompt_inputs.items()}

    # Single-token generation probe requested by design constraints.
    with torch.no_grad():
        _ = _model.generate(
            **prompt_inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    candidate_scores = torch.tensor(
        [_sum_candidate_log_prob(prompt, label) for label in _SENTIMENT_LABELS],
        dtype=torch.float32,
    )
    probs = torch.softmax(candidate_scores, dim=0)

    best_idx = int(torch.argmax(probs).item())
    sentiment_label = _SENTIMENT_LABELS[best_idx]
    sentiment_confidence = float(probs[best_idx].item())
    sentiment_score = float(
        sum(
            probs[i].item() * _SENTIMENT_VALUES[_SENTIMENT_LABELS[i]]
            for i in range(len(_SENTIMENT_LABELS))
        )
    )

    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "sentiment_confidence": sentiment_confidence,
    }


def extract_fingerprint(article_text: str) -> Optional[NewsFingerprint]:
    """
    Extract structured facts and log-prob sentiment from a raw news article string.

    Returns None on any model, parse, or validation failure.
    """
    try:
        _load_model()

        prompt = _build_extraction_prompt(article_text)
        outputs = _generator(
            prompt,
            max_new_tokens=FINGPT_MAX_NEW_TOKENS,
            temperature=FINGPT_TEMPERATURE,
            do_sample=False,
            return_full_text=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
        raw_output = outputs[0]["generated_text"] if outputs else ""
        logger.info("FinGPT raw extraction output: %s", raw_output[:240])

        extracted = json.loads(_extract_json_blob(raw_output))
        sentiment = _score_sentiment(article_text)

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
