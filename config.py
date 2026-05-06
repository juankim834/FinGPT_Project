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
SHARE_SINGLE_LLM_BETWEEN_AGENTS: bool = _env_bool("SHARE_SINGLE_LLM_BETWEEN_AGENTS", True)

# Generation settings for FinGPT fact extraction
FINGPT_MAX_NEW_TOKENS: int = 1024
FINGPT_TEMPERATURE: float = 0.0   # deterministic extraction; no creative generation

# ---------------------------------------------------------------------------
# Anthropic (Agent 2) — Claude
# ---------------------------------------------------------------------------
# ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
# CLAUDE_MODEL: str = "claude-sonnet-4-5"
# CLAUDE_MAX_TOKENS: int = 16_000   # must be > thinking budget
# CLAUDE_THINKING_BUDGET: int = 8_000

# ---------------------------------------------------------------------------
# News API providers
# ---------------------------------------------------------------------------
NEWS_PROVIDER: str = os.getenv("NEWS_PROVIDER", "alpaca").strip().lower()

ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET: str = os.getenv("ALPACA_API_SECRET", "")
ALPACA_NEWS_URL: str = "https://data.alpaca.markets/v1beta1/news"
ALPACA_DEFAULT_LIMIT: int = 50
FINGPT_NEWS_FETCH_COUNT: int = int(os.getenv("FINGPT_NEWS_FETCH_COUNT", str(ALPACA_DEFAULT_LIMIT)))

FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_NEWS_URL: str = "https://finnhub.io/api/v1/company-news"
FINNHUB_TIMEOUT_SEC: float = float(os.getenv("FINNHUB_TIMEOUT_SEC", "15"))
FINNHUB_MAX_CALLS_PER_SEC: float = float(os.getenv("FINNHUB_MAX_CALLS_PER_SEC", "25"))
FINNHUB_MAX_RETRIES: int = int(os.getenv("FINNHUB_MAX_RETRIES", "3"))
FINNHUB_RETRY_BASE_DELAY_SEC: float = float(os.getenv("FINNHUB_RETRY_BASE_DELAY_SEC", "0.5"))

# ---------------------------------------------------------------------------
# Logits-based inference settings
# ---------------------------------------------------------------------------

# Temperature for calibrating softmax probabilities computed from LLM-reported logits.
# Values > 1 soften the distribution (less confident); values < 1 sharpen it.
# Applied after softmax: probs_calibrated = softmax(logits / CALIBRATION_T).
CALIBRATION_T: float = float(os.getenv("FINGPT_CALIBRATION_T", "1.2"))

# Ordered list of sentiment classes for Agent 1.  Order is fixed and must match
# the order the prompt instructs the model to use.
SENTIMENT_CLASSES: list[str] = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

# Ordered list of trading strategies for Agent 2.  Maps index → direction:
#   BUY → long,  HOLD → neutral,  SELL → short.
STRATEGY_SET: list[str] = ["BUY", "HOLD", "SELL"]

# Max tokens for CoT + logits generation (must accommodate <think> block).
LOGITS_MAX_TOKENS: int = int(os.getenv("FINGPT_LOGITS_MAX_TOKENS", "1024"))

# ---------------------------------------------------------------------------
# Logging / output paths
# ---------------------------------------------------------------------------
LOGS_DIR: str = "logs"
OUTPUT_DIR: str = "output"
LOG_LEVEL: str = "INFO"

# ---------------------------------------------------------------------------
# PMI prior cache
# ---------------------------------------------------------------------------
# Path where Agent 2's null-context logprobs (PMI prior) are persisted.
# On first inference the file is written; subsequent sessions load it
# immediately, skipping the extra vLLM call entirely.
# Set to "" to disable persistence (always recompute).
PMI_PRIOR_PATH: str = os.getenv("FINGPT_PMI_PRIOR_PATH", "output/pmi_null_logprobs.json")

# ---------------------------------------------------------------------------
# Agent 1 — event_type logits classifier
# ---------------------------------------------------------------------------

# Score tokens sent to vLLM for event_type classification (A-G, no OTHER).
# OTHER is assigned by Python post-processing when confidence/margin is too low.
EVENT_TYPE_CLASSES: list[str] = ["A", "B", "C", "D", "E", "F", "G"]

# Mapping from score token to concrete event type label.
EVENT_TYPE_MAP: dict[str, str] = {
    "A": "EARNINGS",
    "B": "GUIDANCE",
    "C": "ANALYST_RATING",
    "D": "LEGAL_REGULATORY",
    "E": "MNA",
    "F": "PRODUCT_BUSINESS",
    "G": "MACRO",
}

# Minimum softmax probability for the top event_type class; below this the
# event_type is set to OTHER with method="abstained_low_confidence".
# Default 0.0 = no filtering (all concrete labels accepted).
FINGPT_EVENT_TYPE_MIN_CONFIDENCE: float = float(
    os.getenv("FINGPT_EVENT_TYPE_MIN_CONFIDENCE", "0.0")
)

# Minimum margin (top_prob − second_prob) required to accept a concrete label.
# Below this margin the event_type is set to OTHER with method="abstained_low_margin".
# Default 0.0 = no filtering.
FINGPT_EVENT_TYPE_MIN_MARGIN: float = float(
    os.getenv("FINGPT_EVENT_TYPE_MIN_MARGIN", "0.0")
)

# ---------------------------------------------------------------------------
# Agent 2 — PMI alpha
# ---------------------------------------------------------------------------

# Scaling factor for the PMI null-context correction:
#   adjusted_logit[t] = raw_logit[t] - pmi_alpha * null_logprob[t]
# alpha=1.0 → full PMI correction (original behaviour).
# alpha=0.0 → no PMI correction (raw logits only).
FINGPT_PMI_ALPHA: float = float(os.getenv("FINGPT_PMI_ALPHA", "1.0"))

# ---------------------------------------------------------------------------
# Agent 2 — signal confidence / margin / direction filters
# ---------------------------------------------------------------------------

# If the top-class softmax probability is below this value, force HOLD.
FINGPT_SIGNAL_MIN_CONFIDENCE: float = float(
    os.getenv("FINGPT_SIGNAL_MIN_CONFIDENCE", "0.0")
)

# If the top1−top2 margin is below this value, force HOLD.
FINGPT_SIGNAL_MIN_MARGIN: float = float(
    os.getenv("FINGPT_SIGNAL_MIN_MARGIN", "0.0")
)

# Minimum probability required to emit a BUY signal (when top label is A).
# If prob[A] < threshold, force HOLD.
FINGPT_BUY_THRESHOLD: float = float(os.getenv("FINGPT_BUY_THRESHOLD", "0.0"))

# Minimum probability required to emit a SELL signal (when top label is C).
# If prob[C] < threshold, force HOLD.
FINGPT_SELL_THRESHOLD: float = float(os.getenv("FINGPT_SELL_THRESHOLD", "0.0"))

# ---------------------------------------------------------------------------
# Agent 2 — chain-of-thought mode
# ---------------------------------------------------------------------------

# When True, Agent 2 performs a full CoT generation pass before scoring A/B/C.
# When False (default for backtesting), directly score A/B/C after the compact
# prompt — faster and equally accurate for signal extraction.
FINGPT_SIGNAL_USE_COT: bool = _env_bool("FINGPT_SIGNAL_USE_COT", False)

# ---------------------------------------------------------------------------
# Backtest — strict mode and batch size
# ---------------------------------------------------------------------------

# When True, rows are skipped (not gracefully degraded) on any logits failure:
#   - fingerprint_failed → skipped
#   - event_type_logits_failed → skipped
#   - signal_logits_failed → skipped
# When False (default), graceful fallbacks are applied where available.
FINGPT_BACKTEST_STRICT_MODE: bool = _env_bool("FINGPT_BACKTEST_STRICT_MODE", False)

# Number of articles processed in a single batched vLLM call.
# Overridable via CLI --batch-size.
FINGPT_BACKTEST_BATCH_SIZE: int = int(os.getenv("FINGPT_BACKTEST_BATCH_SIZE", "10"))

# ---------------------------------------------------------------------------
# Backtest - price fetch
# ---------------------------------------------------------------------------

# Number of retries for yfinance price downloads.
FINGPT_PRICE_FETCH_RETRIES: int = int(os.getenv("FINGPT_PRICE_FETCH_RETRIES", "2"))

# Base delay (seconds) between yfinance retries.
FINGPT_PRICE_FETCH_RETRY_BASE_DELAY_SEC: float = float(
    os.getenv("FINGPT_PRICE_FETCH_RETRY_BASE_DELAY_SEC", "0.75")
)

# When False (default), failed price fetches are not persisted to disk cache.
# This avoids "sticky" None caches poisoning later reruns after a transient
# Yahoo/network issue.
FINGPT_PRICE_CACHE_FAILURES: bool = _env_bool("FINGPT_PRICE_CACHE_FAILURES", False)

# Writable directory for yfinance's internal timezone/cache database.
FINGPT_YF_TZ_CACHE_DIR: str = os.getenv(
    "FINGPT_YF_TZ_CACHE_DIR", os.path.join("output", "yfinance_tz_cache")
)
