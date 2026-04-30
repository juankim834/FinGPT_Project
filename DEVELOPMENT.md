# FinGPT Two-Agent Signal Pipeline — Development Guide

## Current Summary

- Both agents now run **locally with HuggingFace** (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`). 
- `agent1` extracts factual `NewsFingerprint` fields and computes sentiment via **token log-prob candidate scoring**.
- `agent2` consumes the fingerprint + Agent 1 sentiment and outputs validated `TradingSignal` JSON with `confidence` and `cot`.
- Added evaluation utilities under `evaluation/`:
  - `collect_batch.py` to build held-out datasets
  - `evaluator.py` for directional accuracy, calibration, and self-consistency
- `notebooks/demo.ipynb` was rewritten for **Google Colab A100** workflow.

---

## 1) Project Overview

This project converts financial news into trading signals using a two-agent pipeline:

1. `ingestion/news_fetcher.py` fetches articles from Alpaca News.
2. `agent1/extractor.py` turns raw article text into a `NewsFingerprint`.
3. `agent2/reasoner.py` turns that fingerprint into a `TradingSignal`.
4. `pipeline.py` orchestrates the flow and saves results to `output/`.

The pipeline is serial, skip-on-failure, and designed for robust batch processing.

---

## 2) Architecture

```
Alpaca News API
   ↓
ingestion/news_fetcher.py
   ↓  list[article dict]
agent1/extractor.py (local HF model)
   ↓  NewsFingerprint
agent2/reasoner.py  (local HF model)
   ↓  TradingSignal
pipeline.py
   ↓
output/signals_<timestamp>.json
```

Evaluation and batch utilities:

```
evaluation/collect_batch.py   -> output/held_out_batch_<timestamp>.json
evaluation/evaluator.py       -> output/evaluation_<timestamp>.json
```

---

## 3) Repository Structure (Key Files)

```
FinGPT_Part2/
├── config.py
├── pipeline.py
├── ingestion/news_fetcher.py
├── agent1/
│   ├── extractor.py
│   ├── prompt.py
│   └── schema.py
├── agent2/
│   ├── reasoner.py
│   ├── prompt.py
│   └── schema.py
├── evaluation/
│   ├── __init__.py
│   ├── collect_batch.py
│   └── evaluator.py
├── notebooks/demo.ipynb
└── backtest/runner.py
```

---

## 4) Data Models

### `agent1/schema.py` — `NewsFingerprint`

```python
class NewsFingerprint(BaseModel):
    source: str
    published_at: str
    headline: str
    companies_named: list[str]
    event_keywords: list[str]
    sentiment_label: Literal[
        "strongly bullish", "bullish", "neutral", "bearish", "strongly bearish"
    ]
    sentiment_score: float         # expected value in [-2, 2]
    sentiment_confidence: float    # max class probability in [0, 1]
```

### `agent2/schema.py` — `TradingSignal`

```python
class TradingSignal(BaseModel):
    ticker: str
    direction: Literal["long", "short", "neutral"]
    strategy_tag: Literal["momentum", "mean_reversion", "event_driven", "macro", "none"]
    confidence: float
    cot: str
```

---

## 5) Agent Details

## Agent 1 (`agent1/extractor.py`)

- Lazy-loads tokenizer/model once (`_load_model`).
- Uses local HF model path/id from `FINGPT_MODEL_PATH`.
- Uses `bfloat16` on CUDA (`float32` on CPU fallback).
- Produces structured extraction + sentiment outputs.
- Sentiment method:
  - fixed label set (5 classes),
  - candidate sequence log-prob scores,
  - softmax probabilities,
  - expected-value score in `[-2, 2]`.

Public API:

```python
extract_fingerprint(article_text: str) -> Optional[NewsFingerprint]
```

## Agent 2 (`agent2/reasoner.py`)

- Fully local HF inference (Anthropic path removed).
- Separate model instance from Agent 1 (separate module-level lazy load).
- Prompt explicitly includes Agent 1 sentiment text.
- Expects strict JSON output and validates via `TradingSignal`.
- Logs reasoning text to `logs/cot_<ticker>_<timestamp>.txt`.

Public API:

```python
generate_signal(fingerprint: NewsFingerprint) -> Optional[TradingSignal]
```

---

## 6) Pipeline Usage

CLI:

```bash
python pipeline.py
python pipeline.py AAPL NVDA TSLA
```

Python:

```python
from pipeline import run_pipeline
signals = run_pipeline(["AAPL", "NVDA"], limit=10)
```

Output:

- `output/signals_<timestamp>.json` (list of `TradingSignal` dicts)

---

## 7) Evaluation Utilities

## `evaluation/collect_batch.py`

Builds held-out datasets by running full pipeline per article:

- Fetches up to 100 Alpaca articles (default ticker basket).
- Runs Agent 1 -> Agent 2 per item.
- Saves successful records to:
  - `output/held_out_batch_<timestamp>.json`
- Logs skip reasons (`Agent 1 failed` / `Agent 2 failed`).

CLI:

```bash
python evaluation/collect_batch.py
```

## `evaluation/evaluator.py`

Provides:

- `evaluate_directional_accuracy(signals)`
- `evaluate_calibration(signals)`
- `evaluate_self_consistency(articles, n_runs=3)`
- `run_full_evaluation(signals_file)`

Saves:

- `output/evaluation_<timestamp>.json`

CLI:

```bash
python -m evaluation.evaluator output/signals_XYZ.json
```

---

## 8) Colab Notebook

`notebooks/demo.ipynb` is now Colab-oriented:

- Declares A100/40GB requirement.
- Installs dependencies in-notebook.
- Loads Alpaca credentials via `google.colab.userdata.get`.
- Uses HF Hub model id (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`).
- Shows VRAM before/after loading both agents.
- Runs fetch -> Agent 1 -> Agent 2 -> DataFrame -> mini self-consistency -> save.
- Uses `tqdm` in loops.

---

## 9) Configuration Notes

`config.py` still contains some legacy Claude constants (`ANTHROPIC_API_KEY`, `CLAUDE_*`) that are not used by the rewritten local Agent 2 path.

Active variables for core flow:

- `FINGPT_MODEL_PATH`
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `LOG_LEVEL`, `OUTPUT_DIR`, `LOGS_DIR`

---

## 10) Known Gaps / Next Improvements

- Add/refresh tests for new Agent 2 local path and evaluation modules.
- Consider adding retry/backoff for Alpaca and yfinance calls.
- Consider stronger JSON schema enforcement in prompts (and optional repair pass).
- Consider caching fetched market data during evaluation to reduce API overhead.
