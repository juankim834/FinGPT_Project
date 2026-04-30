"""
agent1/prompt.py — System prompt for Agent 1 (FinGPT fact extractor).
"""

SYSTEM_PROMPT: str = (
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
