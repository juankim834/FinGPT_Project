from datetime import datetime, timezone
from unittest.mock import patch

from ingestion.news_fetcher import (
    _coerce_timestamp,
    _filter_articles_at_or_before,
    _select_latest_article_per_ticker,
    fetch_recent_articles,
)


def test_coerce_timestamp_handles_iso_z():
    dt = _coerce_timestamp("2026-05-06T12:30:00Z")
    assert dt == datetime(2026, 5, 6, 12, 30, 0, tzinfo=timezone.utc)


def test_filter_articles_at_or_before_is_leakage_safe():
    as_of = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)
    articles = [
        {"headline": "old", "created_at": "2026-05-06T11:59:00Z", "source_ticker": "AAPL"},
        {"headline": "exact", "created_at": "2026-05-06T12:00:00Z", "source_ticker": "AAPL"},
        {"headline": "future", "created_at": "2026-05-06T12:01:00Z", "source_ticker": "AAPL"},
        {"headline": "missing", "created_at": "", "source_ticker": "AAPL"},
    ]

    filtered = _filter_articles_at_or_before(articles, as_of)

    assert [article["headline"] for article in filtered] == ["old", "exact"]


def test_select_latest_article_per_ticker_keeps_nearest_prior_article():
    articles = [
        {"headline": "older aapl", "created_at": "2026-05-06T11:00:00Z", "source_ticker": "AAPL"},
        {"headline": "latest aapl", "created_at": "2026-05-06T11:50:00Z", "source_ticker": "AAPL"},
        {"headline": "latest msft", "created_at": "2026-05-06T11:40:00Z", "source_ticker": "MSFT"},
        {"headline": "older msft", "created_at": "2026-05-06T10:00:00Z", "source_ticker": "MSFT"},
    ]

    selected = _select_latest_article_per_ticker(articles, ["AAPL", "MSFT"])

    assert [article["headline"] for article in selected] == ["latest aapl", "latest msft"]


def test_fetch_recent_articles_filters_and_selects_latest_per_ticker():
    mock_articles = [
        {"headline": "future aapl", "created_at": "2026-05-06T12:30:00Z", "source_ticker": "AAPL"},
        {"headline": "latest aapl", "created_at": "2026-05-06T11:55:00Z", "source_ticker": "AAPL"},
        {"headline": "older aapl", "created_at": "2026-05-06T10:30:00Z", "source_ticker": "AAPL"},
        {"headline": "latest msft", "created_at": "2026-05-06T11:50:00Z", "source_ticker": "MSFT"},
        {"headline": "missing msft ts", "created_at": "", "source_ticker": "MSFT"},
    ]
    as_of = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)

    with patch("ingestion.news_fetcher.NEWS_PROVIDER", "alpaca"), patch(
        "ingestion.news_fetcher._fetch_recent_articles_alpaca",
        return_value=mock_articles,
    ):
        selected = fetch_recent_articles(
            ["AAPL", "MSFT"],
            limit=20,
            as_of_timestamp=as_of,
            nearest_per_ticker=True,
        )

    assert [article["headline"] for article in selected] == ["latest aapl", "latest msft"]
