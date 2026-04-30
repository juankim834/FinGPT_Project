"""
config.py — Global constants for the FinGPT two-agent signal pipeline.
All secrets are read from environment variables; nothing is hardcoded here.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable with common truthy values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# FinGPT (Agent 1) — HuggingFace model
# ---------------------------------------------------------------------------
FINGPT_MODEL_PATH: str = os.getenv("FINGPT_MODEL_PATH", "")
FINGPT_ADAPTER_PATH: str = os.getenv("FINGPT_ADAPTER_PATH", "")
SHARE_SINGLE_LLM_BETWEEN_AGENTS: bool = _env_bool("SHARE_SINGLE_LLM_BETWEEN_AGENTS", False)

# Generation settings for FinGPT fact extraction
FINGPT_MAX_NEW_TOKENS: int = 512
FINGPT_TEMPERATURE: float = 0.0   # deterministic extraction; no creative generation

# ---------------------------------------------------------------------------
# Anthropic (Agent 2) — Claude
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = "claude-sonnet-4-5"
CLAUDE_MAX_TOKENS: int = 16_000   # must be > thinking budget
CLAUDE_THINKING_BUDGET: int = 8_000

# ---------------------------------------------------------------------------
# News API providers
# ---------------------------------------------------------------------------
NEWS_PROVIDER: str = os.getenv("NEWS_PROVIDER", "alpaca").strip().lower()

ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET: str = os.getenv("ALPACA_API_SECRET", "")
ALPACA_NEWS_URL: str = "https://data.alpaca.markets/v1beta1/news"
ALPACA_DEFAULT_LIMIT: int = 50

FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_NEWS_URL: str = "https://finnhub.io/api/v1/company-news"

# ---------------------------------------------------------------------------
# Logging / output paths
# ---------------------------------------------------------------------------
LOGS_DIR: str = "logs"
OUTPUT_DIR: str = "output"
LOG_LEVEL: str = "INFO"
