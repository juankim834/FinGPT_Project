import json
from datetime import date
from pathlib import Path
from uuid import uuid4

from backtest.alpaca_news_pipeline import (
    _config_fingerprint,
    _next_us_trading_day,
    _next_or_same_us_trading_day,
    build_alpaca_backtest_dataset,
    load_alpaca_backtest_config,
    load_or_fetch_alpaca_news,
)


def _make_case_dir() -> Path:
    base = Path("output/test_artifacts") / f"alpaca_pipeline_{uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_config(base_dir: Path, **overrides) -> str:
    payload = {
        "symbols": ["AAPL", "MSFT"],
        "start": "2024-01-01",
        "end": "2024-01-14",
        "frequency": "weekly",
        "fetch_articles_per_symbol": 4,
        "combine_articles_per_sample": 2,
        "holding_period_days": 7,
        "limit_per_request": 50,
        "include_content": True,
        "exclude_contentless": False,
        "content_max_chars": 12,
        "requests_per_minute": 180,
        "cache_dir": "cache",
        "dataset_output_path": "dataset.csv",
        "backtest_output_path": "results.csv",
        "force_refresh": False,
    }
    payload.update(overrides)
    config_path = base_dir / "alpaca_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(config_path)


def test_load_or_fetch_alpaca_news_uses_cache_when_config_unchanged():
    case_dir = _make_case_dir()
    config_path = _write_config(case_dir)
    config = load_alpaca_backtest_config(config_path)

    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    articles = {
        "AAPL": {
            "2024-01-01__2024-01-07": [
                {"headline": "cached", "created_at": "2024-01-02T12:00:00Z", "symbols": ["AAPL"]}
            ],
            "2024-01-08__2024-01-14": [],
        }
    }
    (cache_dir / "news_cache.json").write_text(
        json.dumps({"articles_by_symbol": articles}),
        encoding="utf-8",
    )
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "config_fingerprint": _config_fingerprint(config),
                "article_count": 1,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_or_fetch_alpaca_news(config)
    assert loaded == articles


def test_build_alpaca_backtest_dataset_groups_articles_by_week_and_marks_empty_windows():
    case_dir = _make_case_dir()
    config_path = _write_config(case_dir)
    config = load_alpaca_backtest_config(config_path)
    articles_by_symbol = {
        "AAPL": {
            "2024-01-01__2024-01-07": [
                {
                    "id": 1,
                    "headline": "AI launch",
                    "summary": "summary text",
                    "content": "abcdefghijklmnop",
                    "created_at": "2024-01-03T14:30:00Z",
                    "symbols": ["AAPL"],
                },
                {
                    "id": 2,
                    "headline": "AI follow-up",
                    "summary": "second summary",
                    "content": "",
                    "created_at": "2024-01-04T14:30:00Z",
                    "symbols": ["AAPL"],
                },
            ],
            "2024-01-08__2024-01-14": [],
        },
        "MSFT": {
            "2024-01-01__2024-01-07": [],
            "2024-01-08__2024-01-14": [],
        },
    }

    df = build_alpaca_backtest_dataset(articles_by_symbol, config)

    assert list(df["ticker"]) == ["AAPL", "AAPL", "MSFT", "MSFT"]
    assert bool(df.iloc[0]["skip_llm"]) is False
    assert bool(df.iloc[1]["skip_llm"]) is True
    assert bool(df.iloc[2]["skip_llm"]) is True
    assert bool(df.iloc[3]["skip_llm"]) is True
    assert df.iloc[1]["forced_signal"] == "no_signal"
    assert df.iloc[1]["skip_reason"] == "no_article_provided"
    assert df.iloc[1]["pass_reason"] == "no_article_provided"
    assert "From 2024-01-08 to 2024-01-16" in df.iloc[0]["input"]
    assert "Ticker: AAPL" in df.iloc[0]["input"]
    assert "[Headline]: AI launch" in df.iloc[0]["input"]
    assert "[Headline]: AI follow-up" in df.iloc[0]["input"]
    assert "abcdefghijkl" in df.iloc[0]["input"]
    assert "abcdefghijklm" not in df.iloc[0]["input"]
    assert df.iloc[0]["window_key"] == "2024-01-01__2024-01-07"
    assert df.iloc[1]["window_key"] == "2024-01-08__2024-01-14"


def test_next_us_trading_day_skips_weekends_and_mlk_holiday():
    assert _next_us_trading_day(date(2024, 1, 5)).isoformat() == "2024-01-08"
    assert _next_us_trading_day(date(2024, 1, 14)).isoformat() == "2024-01-16"


def test_next_or_same_us_trading_day_keeps_trading_day_and_skips_holiday():
    assert _next_or_same_us_trading_day(date(2026, 1, 9)).isoformat() == "2026-01-09"
    assert _next_or_same_us_trading_day(date(2026, 1, 10)).isoformat() == "2026-01-12"
