"""
ingestion/news_fetcher.py — Pulls recent articles from the Alpaca News API.

Each returned dict has keys: headline, summary, source, created_at.
Callers should concatenate headline + summary as the article_text passed
to Agent 1.
"""

import logging
from typing import Any

import requests

from config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_DEFAULT_LIMIT,
    ALPACA_NEWS_URL,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def fetch_recent_articles(
    tickers: list[str],
    limit: int = ALPACA_DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """
    Returns a list of article dicts with keys:
        headline, summary, source, created_at

    Concatenate headline + summary as the article_text passed to Agent 1.

    Raises:
        requests.HTTPError: if the Alpaca API returns a non-2xx status.
        ValueError: if API credentials are missing.
    """
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise ValueError(
            "Alpaca credentials not set. "
            "Ensure ALPACA_API_KEY and ALPACA_API_SECRET are in your .env file."
        )

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    params: dict[str, Any] = {
        "symbols": ",".join(tickers),
        "limit": limit,
        "sort": "desc",
    }

    logger.info(
        "Fetching up to %d articles for tickers: %s", limit, tickers
    )

    response = requests.get(ALPACA_NEWS_URL, headers=headers, params=params, timeout=15)
    response.raise_for_status()

    raw_news: list[dict[str, Any]] = response.json().get("news", [])

    articles: list[dict[str, Any]] = [
        {
            "headline": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "created_at": item.get("created_at", ""),
        }
        for item in raw_news
    ]

    logger.info("Fetched %d articles.", len(articles))
    return articles
