"""
agent1/prompt.py — System prompt for Agent 1 (FinGPT fact extractor).
"""

SYSTEM_PROMPT: str = (
    "Extract structured facts from the article below. "
    "Use only information explicitly stated. "
    "Return JSON with these fields:\n"
    "- source: publisher name or empty string\n"
    "- published_at: ISO datetime if present, or empty string\n"
    "- headline: headline exactly as written\n"
    "- companies_named: list of company names or tickers\n"
    "- event_keywords: list of lowercase topic keywords"
)
