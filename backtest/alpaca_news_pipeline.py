"""
Config-driven Alpaca news ingestion pipeline for backtesting.

This module fetches Alpaca news over weekly windows, caches the normalized
articles together with a config fingerprint, materializes a dataset compatible
with the existing news-to-signal backtester, and can optionally execute the
backtest immediately.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from backtest.backtester import run_backtest
from config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_NEWS_URL, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlpacaBacktestConfig:
    symbols: list[str]
    start: str
    end: str
    frequency: str = "weekly"
    fetch_articles_per_symbol: int = 50
    combine_articles_per_sample: int = 5
    holding_period_days: int = 7
    sort: str = "desc"
    limit_per_request: int = 50
    include_content: bool = False
    exclude_contentless: bool = False
    content_max_chars: int = 0
    requests_per_minute: int = 180
    request_timeout_sec: float = 20.0
    max_retries: int = 3
    retry_base_delay_sec: float = 1.0
    cache_dir: str = "output/alpaca_news_cache"
    dataset_output_path: str = "output/alpaca_news_backtest_dataset.csv"
    backtest_output_path: str = "output/alpaca_news_backtest_results.csv"
    batch_size: Optional[int] = None
    max_rows: Optional[int] = None
    force_refresh: bool = False


def _normalize_symbols(symbols: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        text = str(symbol).strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _coerce_rfc3339_or_date(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("Date value cannot be empty.")
    if len(text) == 10:
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_config_file(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return payload


def _resolve_path(base_dir: str, candidate: str) -> str:
    path = Path(candidate)
    if path.is_absolute():
        return str(path)
    return str((Path(base_dir) / path).resolve())


def load_alpaca_backtest_config(config_path: str) -> AlpacaBacktestConfig:
    raw = _load_config_file(config_path)
    base_dir = str(Path(config_path).resolve().parent)
    symbols = _normalize_symbols(list(raw.get("symbols", [])))
    if not symbols:
        raise ValueError("Config must include a non-empty symbols list.")

    config = AlpacaBacktestConfig(
        symbols=symbols,
        start=str(raw.get("start", "")).strip(),
        end=str(raw.get("end", "")).strip(),
        frequency=str(raw.get("frequency", "weekly")).strip().lower(),
        fetch_articles_per_symbol=int(raw.get("fetch_articles_per_symbol", 50)),
        combine_articles_per_sample=int(raw.get("combine_articles_per_sample", 5)),
        holding_period_days=int(raw.get("holding_period_days", 7)),
        sort=str(raw.get("sort", "desc")).strip().lower(),
        limit_per_request=int(raw.get("limit_per_request", 50)),
        include_content=bool(raw.get("include_content", False)),
        exclude_contentless=bool(raw.get("exclude_contentless", False)),
        content_max_chars=int(raw.get("content_max_chars", 0)),
        requests_per_minute=int(raw.get("requests_per_minute", 180)),
        request_timeout_sec=float(raw.get("request_timeout_sec", 20.0)),
        max_retries=int(raw.get("max_retries", 3)),
        retry_base_delay_sec=float(raw.get("retry_base_delay_sec", 1.0)),
        cache_dir=_resolve_path(base_dir, str(raw.get("cache_dir", "output/alpaca_news_cache"))),
        dataset_output_path=_resolve_path(
            base_dir,
            str(raw.get("dataset_output_path", "output/alpaca_news_backtest_dataset.csv")),
        ),
        backtest_output_path=_resolve_path(
            base_dir,
            str(raw.get("backtest_output_path", "output/alpaca_news_backtest_results.csv")),
        ),
        batch_size=(int(raw["batch_size"]) if raw.get("batch_size") is not None else None),
        max_rows=(int(raw["max_rows"]) if raw.get("max_rows") is not None else None),
        force_refresh=bool(raw.get("force_refresh", False)),
    )

    if not config.start or not config.end:
        raise ValueError("Config must include both start and end.")
    if config.frequency != "weekly":
        raise ValueError("frequency currently only supports 'weekly'.")
    if config.sort not in {"asc", "desc"}:
        raise ValueError("sort must be 'asc' or 'desc'.")
    if not 1 <= config.limit_per_request <= 50:
        raise ValueError("limit_per_request must be between 1 and 50.")
    if config.fetch_articles_per_symbol <= 0:
        raise ValueError("fetch_articles_per_symbol must be positive.")
    if config.combine_articles_per_sample <= 0:
        raise ValueError("combine_articles_per_sample must be positive.")
    if config.holding_period_days <= 0:
        raise ValueError("holding_period_days must be positive.")
    if config.requests_per_minute <= 0:
        raise ValueError("requests_per_minute must be positive.")

    start_dt = _coerce_rfc3339_or_date(config.start)
    end_dt = _coerce_rfc3339_or_date(config.end)
    if end_dt < start_dt:
        raise ValueError("end must be on or after start.")
    return config


def _config_fingerprint(config: AlpacaBacktestConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _news_cache_paths(config: AlpacaBacktestConfig) -> tuple[str, str]:
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        str(cache_dir / "news_cache.json"),
        str(cache_dir / "cache_manifest.json"),
    )


def _load_cached_articles(
    config: AlpacaBacktestConfig,
) -> Optional[dict[str, dict[str, list[dict[str, Any]]]]]:
    cache_path, manifest_path = _news_cache_paths(config)
    if not os.path.exists(cache_path) or not os.path.exists(manifest_path):
        return None

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("config_fingerprint") != _config_fingerprint(config):
            logger.info("Config changed since last cache build. Refreshing Alpaca news cache.")
            return None

        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        articles_by_symbol = payload.get("articles_by_symbol", {})
        if not isinstance(articles_by_symbol, dict):
            return None
        logger.info(
            "Loaded cached Alpaca news for %d symbol(s) from %s",
            len(articles_by_symbol),
            cache_path,
        )
        return articles_by_symbol
    except Exception as exc:
        logger.warning("Failed to load Alpaca news cache: %s", exc)
        return None


def _save_cached_articles(
    config: AlpacaBacktestConfig,
    articles_by_symbol: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    cache_path, manifest_path = _news_cache_paths(config)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump({"articles_by_symbol": articles_by_symbol}, fh, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config_fingerprint": _config_fingerprint(config),
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "article_count": sum(
                    len(items)
                    for symbol_windows in articles_by_symbol.values()
                    for items in symbol_windows.values()
                ),
            },
            fh,
            indent=2,
        )
    logger.info(
        "Saved Alpaca cache with %d total article(s) to %s",
        sum(
            len(items)
            for symbol_windows in articles_by_symbol.values()
            for items in symbol_windows.values()
        ),
        cache_path,
    )


class _RateLimiter:
    def __init__(self, requests_per_minute: int):
        self._min_interval = 60.0 / float(requests_per_minute)
        self._last_request_at: Optional[float] = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()


def _alpaca_get_with_retry(
    headers: dict[str, str],
    params: dict[str, Any],
    config: AlpacaBacktestConfig,
    rate_limiter: _RateLimiter,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, config.max_retries + 2):
        rate_limiter.wait()
        try:
            response = requests.get(
                ALPACA_NEWS_URL,
                headers=headers,
                params=params,
                timeout=config.request_timeout_sec,
            )

            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt <= config.max_retries:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(float(retry_after), config.retry_base_delay_sec)
                        except ValueError:
                            delay = config.retry_base_delay_sec * (2 ** (attempt - 1))
                    else:
                        delay = config.retry_base_delay_sec * (2 ** (attempt - 1))
                    logger.warning(
                        "Alpaca news retry (status=%d, attempt=%d, sleep=%.2fs)",
                        response.status_code,
                        attempt,
                        delay,
                    )
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt <= config.max_retries:
                delay = config.retry_base_delay_sec * (2 ** (attempt - 1))
                logger.warning(
                    "Alpaca news request failed (attempt=%d): %s; retrying in %.2fs",
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected Alpaca retry loop termination.")


def _truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def _iter_week_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    current_start = start
    while current_start <= end:
        current_end = min(current_start + timedelta(days=6, hours=23, minutes=59, seconds=59), end)
        windows.append((current_start, current_end))
        current_start = current_end + timedelta(seconds=1)
    return windows


def _normalize_article(
    item: dict[str, Any],
    config: AlpacaBacktestConfig,
) -> dict[str, Any]:
    symbols = _normalize_symbols(list(item.get("symbols", []) or []))
    content = ""
    if config.include_content:
        content = _truncate_text(str(item.get("content", "") or ""), config.content_max_chars)

    return {
        "id": item.get("id"),
        "headline": str(item.get("headline", "") or "").strip(),
        "summary": str(item.get("summary", "") or "").strip(),
        "content": content,
        "author": str(item.get("author", "") or "").strip(),
        "source": str(item.get("source", "") or "").strip(),
        "url": str(item.get("url", "") or "").strip(),
        "created_at": str(item.get("created_at", "") or "").strip(),
        "updated_at": str(item.get("updated_at", "") or "").strip(),
        "symbols": symbols,
    }


def _fetch_alpaca_news_for_symbol_week(
    symbol: str,
    window_start: datetime,
    window_end: datetime,
    config: AlpacaBacktestConfig,
    headers: dict[str, str],
    rate_limiter: _RateLimiter,
) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    page_token: Optional[str] = None

    while len(articles) < config.fetch_articles_per_symbol:
        remaining = config.fetch_articles_per_symbol - len(articles)
        params: dict[str, Any] = {
            "symbols": symbol,
            "start": _to_rfc3339(window_start),
            "end": _to_rfc3339(window_end),
            "sort": config.sort,
            "limit": min(config.limit_per_request, remaining),
            "include_content": str(config.include_content).lower(),
            "exclude_contentless": str(config.exclude_contentless).lower(),
        }
        if page_token:
            params["page_token"] = page_token

        response = _alpaca_get_with_retry(headers, params, config, rate_limiter)
        payload = response.json()
        raw_news = payload.get("news", [])
        if not isinstance(raw_news, list):
            raise ValueError("Unexpected Alpaca news response: 'news' is not a list.")

        batch = [_normalize_article(item, config) for item in raw_news]
        articles.extend(batch)
        page_token = payload.get("next_page_token") or payload.get("page_token")
        logger.info(
            "[%s][week %s..%s] fetched page with %d article(s); accumulated %d/%d",
            symbol,
            window_start.date().isoformat(),
            window_end.date().isoformat(),
            len(batch),
            len(articles),
            config.fetch_articles_per_symbol,
        )
        if not batch or not page_token:
            break

    return articles[: config.fetch_articles_per_symbol]


def fetch_alpaca_news(config: AlpacaBacktestConfig) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise ValueError(
            "Alpaca credentials not set. Ensure ALPACA_API_KEY and ALPACA_API_SECRET are configured."
        )

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    rate_limiter = _RateLimiter(config.requests_per_minute)
    articles_by_symbol: dict[str, dict[str, list[dict[str, Any]]]] = {}
    start_dt = _coerce_rfc3339_or_date(config.start)
    end_dt = _coerce_rfc3339_or_date(config.end)
    week_windows = _iter_week_windows(start_dt, end_dt)

    for symbol in config.symbols:
        symbol_windows: dict[str, list[dict[str, Any]]] = {}
        for window_start, window_end in week_windows:
            window_key = f"{window_start.date().isoformat()}__{window_end.date().isoformat()}"
            symbol_windows[window_key] = _fetch_alpaca_news_for_symbol_week(
                symbol,
                window_start,
                window_end,
                config,
                headers,
                rate_limiter,
            )
        articles_by_symbol[symbol] = symbol_windows

    return articles_by_symbol


def load_or_fetch_alpaca_news(
    config: AlpacaBacktestConfig,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if not config.force_refresh:
        cached = _load_cached_articles(config)
        if cached is not None:
            return cached

    articles_by_symbol = fetch_alpaca_news(config)
    _save_cached_articles(config, articles_by_symbol)
    return articles_by_symbol


def _articles_to_prompt(
    articles: list[dict[str, Any]],
    symbol: str,
    window_key: str,
    window_end: datetime,
    holding_period_days: int,
    content_max_chars: int,
) -> dict[str, Any]:
    created_at_values = [_coerce_rfc3339_or_date(article["created_at"]) for article in articles]
    start_dt = min(created_at_values)
    start_date = start_dt.date().isoformat()
    end_date = (window_end.date() + timedelta(days=holding_period_days)).isoformat()

    segments: list[str] = []
    for article in articles:
        headline = article.get("headline", "").strip()
        summary_parts = [article.get("summary", "").strip()]
        content = _truncate_text(article.get("content", "").strip(), content_max_chars)
        if content:
            summary_parts.append(content)
        summary_text = "\n\n".join(part for part in summary_parts if part).strip()
        segments.append(f"[Headline]: {headline}\n[Summary]: {summary_text}")

    input_text = (
        f"From {start_date} to {end_date}\n"
        f"{chr(10).join(segments)}\n"
        f"[Basic Financials]: N/A"
    )

    return {
        "input": input_text,
        "output": "",
        "answer": "",
        "ticker": symbol,
        "window_key": window_key,
        "skip_llm": False,
        "forced_signal": "",
        "skip_reason": "",
        "pass_reason": "",
    }


def _skip_row_for_symbol(symbol: str, window_key: str) -> dict[str, Any]:
    return {
        "input": "",
        "output": "",
        "answer": "",
        "ticker": symbol,
        "window_key": window_key,
        "skip_llm": True,
        "forced_signal": "no_signal",
        "skip_reason": "no_article_provided",
        "pass_reason": "no_article_provided",
    }


def build_alpaca_backtest_dataset(
    articles_by_symbol: dict[str, dict[str, list[dict[str, Any]]]],
    config: AlpacaBacktestConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    chunk_size = config.combine_articles_per_sample

    for symbol in config.symbols:
        symbol_windows = articles_by_symbol.get(symbol, {})
        for window_key, window_articles in symbol_windows.items():
            _, window_end_text = window_key.split("__", 1)
            window_end = _coerce_rfc3339_or_date(window_end_text)
            if not window_articles:
                rows.append(_skip_row_for_symbol(symbol, window_key))
                continue

            for idx in range(0, len(window_articles), chunk_size):
                chunk = window_articles[idx : idx + chunk_size]
                if not chunk:
                    continue
                rows.append(
                    _articles_to_prompt(
                        chunk,
                        symbol,
                        window_key,
                        window_end,
                        config.holding_period_days,
                        config.content_max_chars,
                    )
                )

    return pd.DataFrame(
        rows,
        columns=[
            "input",
            "output",
            "answer",
            "ticker",
            "window_key",
            "skip_llm",
            "forced_signal",
            "skip_reason",
            "pass_reason",
        ],
    )


def save_alpaca_backtest_dataset(df: pd.DataFrame, output_path: str) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Saved Alpaca backtest dataset with %d row(s) to %s", len(df), output)
    return str(output)


def run_alpaca_backtest_pipeline(
    config_path: str,
    *,
    run_existing_backtest: bool = True,
) -> dict[str, Any]:
    config = load_alpaca_backtest_config(config_path)
    articles_by_symbol = load_or_fetch_alpaca_news(config)
    dataset = build_alpaca_backtest_dataset(articles_by_symbol, config)
    dataset_path = save_alpaca_backtest_dataset(dataset, config.dataset_output_path)
    article_count = sum(
        len(items)
        for symbol_windows in articles_by_symbol.values()
        for items in symbol_windows.values()
    )

    result: dict[str, Any] = {
        "config": config,
        "article_count": article_count,
        "dataset_rows": int(len(dataset)),
        "dataset_path": dataset_path,
        "backtest_output_path": config.backtest_output_path,
    }

    if run_existing_backtest:
        backtest_df = run_backtest(
            dataset_path=dataset_path,
            output_path=config.backtest_output_path,
            max_rows=config.max_rows,
            batch_size=config.batch_size,
        )
        result["backtest_rows"] = int(len(backtest_df))

    return result
