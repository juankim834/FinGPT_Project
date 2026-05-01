"""
agent2/prompt.py — Prompts for Agent 2 (trading strategy reasoner).

STRATEGY_COT_PROMPT
    Instructs the DeepSeek-R1 model to reason about the appropriate trading
    strategy given the news article and Agent 1's sentiment analysis.
    Reasoning is written inside ``<think>…</think>`` tags; vLLM stops
    generation at ``</think>``.

    After the CoT, the model picks one of three lettered options (A / B / C).
    ``vllm_logits_client.get_real_choice_logits`` scores the single-character
    tokens "A", "B", "C" via ``SamplingParams(prompt_logprobs=1)``, which
    have near-equal language-model priors and therefore avoid the directional
    bias that arises when scoring the words "BUY", "HOLD", "SELL" directly.

    The model is NOT asked to output a JSON object or numeric logits.

STRATEGY_DECISION_PREFIX
    Appended after ``</think>\\n`` in Phase 2 to create the decision context.

    **Must end with a space or a character that cannot BPE-merge with the
    first character of the scoring tokens.**  The previous prefix
    ``"The answer is ("`` caused ``"(A"``, ``"(B"``, ``"(C"`` to be merged
    into single tokens by the LLaMA/DeepSeek BPE vocabulary, making
    ``_resolve_choice_token_ids`` produce an empty ``choice_ids`` list and
    ``_sum_choice_logprob`` silently return ``0.0`` for all choices —
    yielding ``softmax([0,0,0]) = [1/3,1/3,1/3]``.

    ``"Strategy: "`` ends with ``": "`` (colon + space).  The LLaMA/DeepSeek
    tokenizer never merges a trailing space with a following capital letter,
    so "A", "B", "C" are always resolved as distinct tokens.

STRATEGY_SCORE_TOKENS
    The actual tokens passed to ``get_real_choice_logits`` / ``_batch``.
    Single ASCII letters are reliably single-token in all BPE vocabularies and
    have balanced priors, eliminating the HOLD-dominance artefact seen when
    scoring the words "BUY" / "HOLD" / "SELL" directly.
"""

# ---------------------------------------------------------------------------
# Strategy CoT prompt (Phase 1 — reasoning only, no answer expected)
# ---------------------------------------------------------------------------

STRATEGY_COT_PROMPT: str = """\
You are a quantitative trading strategist.

Based on the news article and sentiment analysis below, reason step by step \
about the appropriate one-week trading action. Write ALL of your analysis \
inside <think>...</think> tags. Consider:

• What near-term price direction does this news imply?
• Does the sentiment analysis align with the factual content?
• Is the signal strong enough to act on (A or C), or should we wait (B)?
• How confident are you given any ambiguity in the news?

At the end you will select one lettered option:
  A = BUY  — bullish signal; expect the price to rise over the next week
  B = HOLD — insufficient or ambiguous signal; stay flat
  C = SELL — bearish signal; expect the price to fall over the next week

Sentiment analysis:
{sentiment_json}

News article:
{article_text}\
"""

# ---------------------------------------------------------------------------
# Decision prefix and scoring tokens (Phase 2 — prompt_logprobs scoring)
# ---------------------------------------------------------------------------

# Appended after </think>\n.  Ends with ": " (colon + space) so that single-
# letter tokens A / B / C are never BPE-merged with the preceding character.
# (The previous "The answer is (" caused "(A"/"(B"/"(C" to merge into single
# tokens, producing empty choice_ids and uniform [0,0,0] logprobs.)
STRATEGY_DECISION_PREFIX: str = "Strategy: "

# Tokens passed to get_real_choice_logits / _batch.
# Positional mapping: A → BUY (idx 0), B → HOLD (idx 1), C → SELL (idx 2).
# Must stay aligned with STRATEGY_SET in config.py.
STRATEGY_SCORE_TOKENS: list[str] = ["A", "B", "C"]

# Backward-compatible alias.
SYSTEM_PROMPT: str = STRATEGY_COT_PROMPT
