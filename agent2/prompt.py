"""
agent2/prompt.py — System prompt for Agent 2 trading reasoner.
"""

SYSTEM_PROMPT: str = (
    "Given the news fingerprint and sentiment below, produce a trading signal. "
    "Return only a JSON object.\n"
    "Good output example:\n"
    '{"ticker":"AAPL","direction":"long","strategy_tag":"momentum","confidence":0.74,'
    '"cot":"Recent catalysts and sentiment both support near-term upside. Momentum remains positive with no immediate contrary signal."}\n'
    "Now return JSON with these fields:\n"
    "- ticker: one ticker from companies_named\n"
    "- direction: long, short, or neutral\n"
    "- strategy_tag: momentum, mean_reversion, event_driven, macro, or none\n"
    "- confidence: float between 0 and 1\n"
    "- cot: 2-4 sentence plain-English reasoning"
)
