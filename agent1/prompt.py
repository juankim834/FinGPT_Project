"""
agent1/prompt.py — Prompts for Agent 1.

Two separate prompts are used:

1. EXTRACTION_PROMPT  — instructs the model to extract structured facts from
   the article and return them as a JSON object.  Guided decoding enforces the
   NewsFingerprint schema so no CoT is needed here.

2. SENTIMENT_COT_PROMPT — instructs the DeepSeek-R1 model to:
     Step 1: Reason inside <think>…</think> about the sentiment.
     Step 2: Output a compact JSON object with self-reported logits for the
             three sentiment classes in a fixed order (POSITIVE, NEGATIVE, NEUTRAL).
   All probability calculations (softmax, calibration, argmax) are done
   deterministically in Python after the LLM returns these logits.
"""

# ---------------------------------------------------------------------------
# Fact-extraction prompt (unchanged from previous version)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT: str = (
    "Extract structured facts from the article below. "
    "Use only information explicitly stated. "
    "Return only a JSON object.\n"
    "Good output example:\n"
    '{"source":"Reuters","published_at":"2024-02-25T14:30:00Z","headline":"Apple shares rise after strong iPhone demand",'
    '"companies_named":["Apple","AAPL"],"event_keywords":["iphone","demand","earnings"]}\n'
    "Now return JSON with these fields:\n"
    "- source: publisher name or empty string\n"
    "- published_at: ISO datetime if present, or empty string\n"
    "- headline: main headline; if multiple headlines are provided, write one concise combined headline\n"
    "- companies_named: list of company names or tickers\n"
    "- event_keywords: list of lowercase topic keywords"
)

# Keep old name as alias so any external imports still resolve.
SYSTEM_PROMPT: str = EXTRACTION_PROMPT

# ---------------------------------------------------------------------------
# Sentiment CoT + logits prompt
# ---------------------------------------------------------------------------

SENTIMENT_COT_PROMPT: str = """\
You are a financial news sentiment analyst.

Your task has two steps:

Step 1 — Reasoning (write inside <think> tags):
Analyse the news article carefully.  Consider:
  • Which companies or sectors are affected?
  • Is the news fundamentally positive, negative, or neutral for investors?
  • Are there any conflicting signals or ambiguity?
Write your reasoning inside <think> and </think> tags.

Step 2 — Final output (after the </think> tag):
Output a single JSON object containing the raw logit scores for exactly three
sentiment classes in this fixed order: POSITIVE, NEGATIVE, NEUTRAL.

Rules:
  • Do NOT output a sentiment label, probability, or any text outside the JSON.
  • The logit values are unscaled real numbers (positive or negative).
  • Higher logit → stronger belief in that class.
  • The JSON must have exactly one key: "logits", whose value is a list of
    three numbers.

Few-shot example
----------------
Article: "Company X beats earnings estimates by 15%; CEO raises full-year guidance."

<think>
The article reports a strong earnings beat and positive guidance revision.
This is clearly good news for investors.  Sentiment should be strongly positive.
NEGATIVE and NEUTRAL logits should be much lower.
</think>
{"logits": [2.8, -1.2, 0.1]}

----------------
Now analyse the following article:

{article_text}
"""
