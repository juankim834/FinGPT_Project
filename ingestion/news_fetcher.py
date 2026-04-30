"""
ingestion/news_fetcher.py — Pulls recent articles from Alpaca or Finnhub.

Each returned dict has keys: headline, summary, source, created_at.
Callers should concatenate headline + summary as the article_text passed
to Agent 1.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_DEFAULT_LIMIT,
    ALPACA_NEWS_URL,
    FINNHUB_API_KEY,
    FINNHUB_NEWS_URL,
    LOG_LEVEL,
    NEWS_PROVIDER,
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def _fetch_recent_articles_alpaca(
    tickers: list[str],
    limit: int,
) -> list[dict[str, Any]]:
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

    logger.info("Fetching up to %d Alpaca articles for tickers: %s", limit, tickers)

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

    logger.info("Fetched %d Alpaca articles.", len(articles))
    return articles


def _fetch_recent_articles_finnhub(
    tickers: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not FINNHUB_API_KEY:
        raise ValueError(
            "Finnhub credentials not set. "
            "Ensure FINNHUB_API_KEY is in your .env file."
        )

    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=7)

    articles: list[dict[str, Any]] = []
    for ticker in tickers:
        params: dict[str, Any] = {
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": FINNHUB_API_KEY,
        }
        response = requests.get(FINNHUB_NEWS_URL, params=params, timeout=15)
        response.raise_for_status()
        raw_news: list[dict[str, Any]] = response.json()

        for item in raw_news:
            created_unix = item.get("datetime")
            created_at = ""
            if isinstance(created_unix, (int, float)):
                created_at = datetime.fromtimestamp(created_unix, tz=timezone.utc).isoformat()

            articles.append(
                {
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "created_at": created_at,
                }
            )

    # Keep newest first and cap at requested limit across all tickers.
    articles.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in articles:
        dedupe_key = (item.get("headline", ""), item.get("created_at", ""))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(item)
        if len(deduped) >= limit:
            break

    logger.info("Fetched %d Finnhub articles.", len(deduped))
    return deduped


def fetch_recent_articles(
    tickers: list[str],
    limit: int = ALPACA_DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """
    Returns a list of article dicts with keys:
        headline, summary, source, created_at

    Concatenate headline + summary as the article_text passed to Agent 1.

    Raises:
        requests.HTTPError: if the selected API returns a non-2xx status.
        ValueError: if API credentials are missing.
    """
    provider = NEWS_PROVIDER or "alpaca"
    if provider == "finnhub":
        return _fetch_recent_articles_finnhub(tickers=tickers, limit=limit)
    if provider == "alpaca":
        return _fetch_recent_articles_alpaca(tickers=tickers, limit=limit)
    raise ValueError(
        f"Unsupported NEWS_PROVIDER={provider!r}. Use 'alpaca' or 'finnhub'."
    )
