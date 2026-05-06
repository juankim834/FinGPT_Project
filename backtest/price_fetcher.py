"""
Price fetching helpers for backtesting.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from config import (
    FINGPT_PRICE_CACHE_FAILURES,
    FINGPT_PRICE_FETCH_RETRIES,
    FINGPT_PRICE_FETCH_RETRY_BASE_DELAY_SEC,
    FINGPT_YF_TZ_CACHE_DIR,
)

_RETURN_CACHE: dict[tuple[str, str, str], Optional[float]] = {}
_DISK_CACHE: dict[str, Optional[float]] = {}
_DISK_CACHE_LOADED = False
_CACHE_PATH_ENV = "FINGPT_YF_CACHE_PATH"
_DEFAULT_CACHE_PATH = os.path.join("output", "yfinance_return_cache.json")
_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
logger = logging.getLogger(__name__)


def _configure_yfinance_cache() -> None:
    os.makedirs(FINGPT_YF_TZ_CACHE_DIR, exist_ok=True)
    setter = getattr(yf, "set_tz_cache_location", None)
    if callable(setter):
        setter(FINGPT_YF_TZ_CACHE_DIR)


_configure_yfinance_cache()


def _cache_key(ticker: str, start_date: str, end_date: str) -> str:
    return f"{ticker}|{start_date}|{end_date}"


def _cache_path() -> str:
    return os.getenv(_CACHE_PATH_ENV, _DEFAULT_CACHE_PATH)


def _load_disk_cache() -> None:
    global _DISK_CACHE_LOADED  # noqa: PLW0603
    if _DISK_CACHE_LOADED:
        return
    _DISK_CACHE_LOADED = True

    path = _cache_path()
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key, value in payload.items():
                _DISK_CACHE[str(key)] = float(value) if value is not None else None
    except Exception:
        # Corrupt cache should never block backtest execution.
        _DISK_CACHE.clear()


def _save_disk_cache() -> None:
    path = _cache_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_DISK_CACHE, handle)


def _store_cached_value(
    ticker: str,
    start_date: str,
    end_date: str,
    value: Optional[float],
) -> Optional[float]:
    key = (ticker, start_date, end_date)
    _RETURN_CACHE[key] = value
    if value is not None or FINGPT_PRICE_CACHE_FAILURES:
        _DISK_CACHE[_cache_key(ticker, start_date, end_date)] = value
        _save_disk_cache()
    return value


def _date_to_epoch(date_str: str, *, add_days: int = 0) -> int:
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if add_days:
        base = base + timedelta(days=add_days)
    return int(base.timestamp())


def _extract_chart_closes(payload: dict) -> list[float]:
    chart = payload.get("chart", {})
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return []

    result = results[0]
    indicators = result.get("indicators", {})
    adjclose_list = indicators.get("adjclose")
    if isinstance(adjclose_list, list) and adjclose_list:
        maybe_adjclose = adjclose_list[0].get("adjclose")
        if isinstance(maybe_adjclose, list):
            return [float(v) for v in maybe_adjclose if isinstance(v, (int, float))]

    quote_list = indicators.get("quote")
    if isinstance(quote_list, list) and quote_list:
        maybe_close = quote_list[0].get("close")
        if isinstance(maybe_close, list):
            return [float(v) for v in maybe_close if isinstance(v, (int, float))]

    return []


def _fetch_yahoo_chart_return(
    ticker: str,
    start_date: str,
    end_date: str,
) -> tuple[Optional[float], str]:
    params = {
        "period1": _date_to_epoch(start_date),
        "period2": _date_to_epoch(end_date, add_days=1),
        "interval": "1d",
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    url = _YAHOO_CHART_URL.format(ticker=ticker)

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.warning(
            "Yahoo chart request failed for %s %s->%s: %s",
            ticker,
            start_date,
            end_date,
            exc,
        )
        return None, "chart_request_failed"
    except ValueError:
        return None, "chart_bad_json"

    closes = _extract_chart_closes(payload)
    if not closes:
        return None, "empty_history"

    first_close = closes[0]
    last_close = closes[-1]
    if first_close == 0.0:
        return None, "zero_first_close"

    return (last_close - first_close) / first_close, ""


def _fetch_yfinance_return(
    ticker: str,
    start_date: str,
    end_date: str,
) -> tuple[Optional[float], str]:
    try:
        hist = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.warning(
            "yfinance download failed for %s %s->%s: %s",
            ticker,
            start_date,
            end_date,
            exc,
        )
        return None, "download_exception"

    if hist is None or hist.empty:
        return None, "empty_history"

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    if "Close" not in hist.columns:
        return None, "missing_close"

    closes = hist["Close"].dropna()
    if closes.empty:
        return None, "empty_closes"

    first_close = float(closes.iloc[0])
    last_close = float(closes.iloc[-1])
    if first_close == 0.0:
        return None, "zero_first_close"

    return (last_close - first_close) / first_close, ""


def clear_price_cache(*, clear_disk: bool = False) -> None:
    """
    Clear in-memory cache and optionally remove the persisted cache file.
    """
    _RETURN_CACHE.clear()
    global _DISK_CACHE_LOADED  # noqa: PLW0603
    if clear_disk:
        _DISK_CACHE.clear()
        path = _cache_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("Failed to remove yfinance cache file: %s", path)
    _DISK_CACHE_LOADED = False


def get_realized_return_with_reason(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    refresh: bool = False,
) -> tuple[Optional[float], str]:
    """
    Fetch daily adjusted closes and compute close-to-close realized return.

    Parameters
    ----------
    refresh:
        When True, bypass any cached value and fetch again. This is useful for
        recovering a backtest after a transient Yahoo/network outage.

    Returns
    -------
    tuple[Optional[float], str]
        `(realized_return, reason)` where `reason == ""` means success.
    """
    _load_disk_cache()
    key = (ticker, start_date, end_date)
    if not refresh and key in _RETURN_CACHE:
        cached = _RETURN_CACHE[key]
        return cached, "" if cached is not None else "cached_failure"

    disk_key = _cache_key(ticker, start_date, end_date)
    if not refresh and disk_key in _DISK_CACHE:
        cached = _DISK_CACHE[disk_key]
        _RETURN_CACHE[key] = cached
        return cached, "" if cached is not None else "cached_failure"

    if not ticker or not start_date or not end_date:
        return _store_cached_value(ticker, start_date, end_date, None), "missing_inputs"

    last_reason = "unknown"
    attempts = FINGPT_PRICE_FETCH_RETRIES + 1
    for attempt in range(attempts):
        realized_return, last_reason = _fetch_yahoo_chart_return(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        if realized_return is not None:
            return _store_cached_value(ticker, start_date, end_date, realized_return), ""

        realized_return, yfinance_reason = _fetch_yfinance_return(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        if realized_return is not None:
            return _store_cached_value(ticker, start_date, end_date, realized_return), ""
        last_reason = yfinance_reason or last_reason

        if attempt < attempts - 1:
            time.sleep(FINGPT_PRICE_FETCH_RETRY_BASE_DELAY_SEC * (attempt + 1))

    return _store_cached_value(ticker, start_date, end_date, None), last_reason


def get_realized_return(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    refresh: bool = False,
) -> Optional[float]:
    """
    Backward-compatible wrapper that returns only the realized return.
    """
    realized_return, _ = get_realized_return_with_reason(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    return realized_return


def direction_from_return(realized_return: float, threshold: float = 0.001) -> str:
    """
    Convert realized return into up/down/neutral direction label.
    """
    if realized_return > threshold:
        return "up"
    if realized_return < -threshold:
        return "down"
    return "neutral"


def get_cache_stats() -> dict[str, int]:
    """
    Return simple stats for in-memory + disk yfinance return cache.
    """
    _load_disk_cache()
    disk_entries = len(_DISK_CACHE)
    memory_entries = len(_RETURN_CACHE)
    return {
        "disk_entries": disk_entries,
        "memory_entries": memory_entries,
    }
