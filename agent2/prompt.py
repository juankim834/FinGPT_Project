"""
agent2/prompt.py — Prompts for Agent 2 (trading direction classifier).

Agent 2 is a narrow direction classifier. It receives only the structured
Agent 1 fingerprint:
  - ticker and headline
  - sentiment label + probabilities
  - event_type + confidence/margin/method
  - companies_named

It does NOT re-read the full article text, re-extract event keywords,
predict strategy_tag, or report self-assessed confidence.  All of those
are either handled by Agent 1 or set deterministically.

STRATEGY_COT_PROMPT
    Compact prompt presenting the ticker, headline, and Agent 1 fingerprint.

    Two variants are supported via FINGPT_SIGNAL_USE_COT:
      - No-CoT (default): prompt ends immediately before the decision prefix;
        A/B/C logprobs are read directly without a generation pass.
      - CoT mode: a <think>...</think> block is requested before the answer,
        matching the DeepSeek-R1 reasoning format.

    In both cases the actual decision scoring uses single-character tokens
    A / B / C via prompt_logprobs, avoiding BPE-merge issues and label-prior
    imbalance that arise when scoring "BUY" / "HOLD" / "SELL" directly.

STRATEGY_DECISION_PREFIX
    Appended after the prompt (no-CoT) or after ``</think>\\n`` (CoT mode).
    Ends with ``": "`` (colon + space) so A / B / C tokens are never
    BPE-merged with the preceding character.

STRATEGY_SCORE_TOKENS
    The actual tokens passed to ``get_real_choice_logits`` / ``_batch``.
    Positional mapping: A → BUY (idx 0), B → HOLD (idx 1), C → SELL (idx 2).
    Must stay aligned with STRATEGY_SET in config.py.
"""

# ---------------------------------------------------------------------------
# Strategy prompt — compact, fingerprint-only (no article text re-read)
# ---------------------------------------------------------------------------

# No-CoT variant: used when FINGPT_SIGNAL_USE_COT=False (default).
# The model scores A/B/C directly after this prompt + decision prefix.
STRATEGY_PROMPT_NO_COT: str =  """\
You are a conservative financial trading signal classifier.

Given the structured news fingerprint, choose one trading direction for the target ticker.

Target ticker: {ticker}
Headline: {headline}

Structured fingerprint:
- Sentiment label: {sentiment_label}
- Sentiment confidence: {sentiment_confidence}
- Sentiment probabilities:
  - positive: {p_pos}
  - neutral: {p_neu}
  - negative: {p_neg}
- Event type: {event_type}
- Event type confidence: {event_type_confidence}
- Event type margin: {event_type_margin}

Decision rules:
- Choose A only when the news is clearly positive for the target ticker.
- Choose C only when the news is clearly negative for the target ticker and the negative sentiment is high-confidence.
- Choose B when the signal is mixed, weak, macro-only, unclear, low-confidence, or not clearly tradable.
- Do not infer beyond the given fingerprint.
- Prefer B over A or C when uncertain.

Directions:
A = BUY / long
B = HOLD / neutral
C = SELL / short

At "Strategy:", choose only one letter: A, B, or C.\
"""

# CoT variant: used when FINGPT_SIGNAL_USE_COT=True.
# Adds a <think>...</think> reasoning block; vLLM stops at </think>.
STRATEGY_PROMPT_COT: str = """\
You are a financial trading signal classifier.

Given the structured news fingerprint, reason step by step \
about the appropriate one-week trading direction. Write ALL of your \
reasoning inside <think>...</think> tags. Then choose one direction.

Directions:
A = BUY / long
B = HOLD / neutral
C = SELL / short

Ticker: {ticker}
Headline: {headline}

Agent 1 fingerprint:
- Sentiment: {sentiment_label}
- Sentiment confidence: {sentiment_confidence}
- Sentiment probabilities: positive={p_pos}, neutral={p_neu}, negative={p_neg}
- Event type: {event_type}
- Event type confidence: {event_type_confidence}
- Event type margin: {event_type_margin}
- Event type method: {event_type_method}
- Companies named: {companies_named}\
"""

# Default: no-CoT for backward compatibility and backtest speed.
# Callers should use STRATEGY_PROMPT_COT when FINGPT_SIGNAL_USE_COT=True.
STRATEGY_COT_PROMPT: str = STRATEGY_PROMPT_NO_COT

# ---------------------------------------------------------------------------
# Decision prefix and scoring tokens (Phase 2 — prompt_logprobs scoring)
# ---------------------------------------------------------------------------

# Appended after the prompt (no-CoT) or after </think>\n (CoT mode).
# Ends with ": " (colon + space) so that single-letter tokens A / B / C
# are never BPE-merged with the preceding character.
STRATEGY_DECISION_PREFIX: str = "Strategy: "

# Tokens passed to get_real_choice_logits / _batch.
# Positional mapping: A → BUY (idx 0), B → HOLD (idx 1), C → SELL (idx 2).
# Must stay aligned with STRATEGY_SET in config.py.
STRATEGY_SCORE_TOKENS: list[str] = ["A", "B", "C"]

# Backward-compatible alias.
SYSTEM_PROMPT: str = STRATEGY_COT_PROMPT
