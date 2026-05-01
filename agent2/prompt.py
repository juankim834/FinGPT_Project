"""
agent2/prompt.py — Prompts for Agent 2 (trading strategy reasoner).

STRATEGY_COT_PROMPT
    Instructs the DeepSeek-R1 model to reason about the appropriate trading
    strategy given the news article and Agent 1's sentiment analysis.
    Reasoning is written inside ``<think>…</think>`` tags; vLLM stops
    generation at ``</think>``.

    ``vllm_logits_client.get_real_choice_logits`` then scores each strategy
    label (BUY, HOLD, SELL) by reading real token-level log-probabilities via
    ``SamplingParams(prompt_logprobs=1)`` at the decision point.

    The model is NOT asked to output a JSON object or numeric logits.

STRATEGY_DECISION_PREFIX
    Appended after the CoT to create the decision context for scoring.
    Must end with a space so the choice token attaches cleanly.
"""

# ---------------------------------------------------------------------------
# Strategy CoT prompt (Phase 1 — reasoning only, no answer expected)
# ---------------------------------------------------------------------------

STRATEGY_COT_PROMPT: str = """\
You are a quantitative trading strategist.

Based on the news article and sentiment analysis below, reason step by step \
about the appropriate trading strategy. Write your analysis inside \
<think>...</think> tags. Consider:

• What price direction does this news imply in the near term?
• Does the sentiment analysis align with the factual content?
• Is the signal strong enough to trade (BUY or SELL), or should we wait (HOLD)?
• How confident are you given any ambiguity in the news?

Sentiment analysis:
{sentiment_json}

News article:
{article_text}\
"""

# ---------------------------------------------------------------------------
# Decision prefix (Phase 2 — appended after </think> for logprob scoring)
# ---------------------------------------------------------------------------

# The model's next-token distribution at "Strategy: " is scored for each
# of the three strategy labels: BUY, HOLD, SELL.
# Must end with a space so the choice token aligns with BPE tokenisation.
STRATEGY_DECISION_PREFIX: str = "Strategy: "

# Backward-compatible alias.
SYSTEM_PROMPT: str = STRATEGY_COT_PROMPT
