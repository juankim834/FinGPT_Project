# FinGPT Part 2 Development Guide

This document summarizes the current repository as it exists today and is intended to help development work move quickly without re-discovering the architecture.

## Repo Purpose

This project implements a local, inference-only financial news pipeline:

1. `Agent 1` extracts a structured `NewsFingerprint` from article text — including guided JSON fact extraction, logits-based sentiment classification, and logits-based event type classification.
2. `Agent 2` receives only the ticker, headline, and Agent 1 structured fingerprint — and predicts a trading direction (BUY/HOLD/SELL) using A/B/C token logprobs.
3. The `backtest` package runs the same pipeline over a labeled dataset and evaluates realized returns.

The core design choice is that decisions are derived from real token log-probabilities read from vLLM, not from self-reported numbers produced by the model.

## High-Level Architecture

### Main flow

- `pipeline.py`
  - Live entry point.
  - Fetches articles from Alpaca or Finnhub.
  - Concatenates `headline + summary`.
  - Runs `extract_fingerprint()` then `generate_signal()`.
  - Saves results to `output/signals_<timestamp>.json`.

### Agent 1

- [agent1/extractor.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/agent1/extractor.py)
  - Loads or reuses a local vLLM engine.
  - Step 1: Guided decoding for structured JSON fact extraction (source, headline, companies_named).
  - Step 2: Logits-based sentiment classification (POSITIVE / NEGATIVE / NEUTRAL) via CoT + prompt_logprobs.
  - Step 3: Logits-based event type classification (A-G → EARNINGS / GUIDANCE / ANALYST_RATING / LEGAL_REGULATORY / MNA / PRODUCT_BUSINESS / MACRO) via direct prompt_logprobs (no CoT).
  - OTHER is assigned by Python post-processing when confidence or margin is below threshold — it is never a model-scored token.
  - Shared helpers: `_rank_probabilities`, `_apply_classification_thresholds`.
  - Ticker fallback: if `companies_named` is empty and a ticker is available from the dataset, it is substituted.
- [agent1/schema.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/agent1/schema.py)
  - Defines `NewsFingerprint`.
- [agent1/prompt.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/agent1/prompt.py)
  - Extraction prompt, sentiment CoT/scoring prompts, event type classification prompt.

### Agent 2

- [agent2/reasoner.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/agent2/reasoner.py)
  - Reuses Agent 1's vLLM engine when `SHARE_SINGLE_LLM_BETWEEN_AGENTS=true`.
  - Receives only: ticker, headline, Agent 1 sentiment fields, Agent 1 event_type fields, companies_named.
  - Does NOT receive the full article text, event keywords, or strategy tag prediction.
  - Scores A/B/C logprobs (BUY/HOLD/SELL) using prompt_logprobs.
  - Applies PMI correction: `adjusted = raw - pmi_alpha * null_logprobs`.
  - Applies confidence/margin/direction threshold filters that can force HOLD.
  - strategy_tag is always set to "event_driven" — it is not predicted by the model.
  - Optional CoT mode (FINGPT_SIGNAL_USE_COT) for qualitative debugging.
- [agent2/schema.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/agent2/schema.py)
  - Defines `TradingSignal`.
- [agent2/prompt.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/agent2/prompt.py)
  - Compact direction-classification prompt (no-CoT and CoT variants), decision prefix, score tokens.

### Shared logits helper

- [vllm_logits_client.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/vllm_logits_client.py)
  - Encapsulates the two-phase vLLM pattern:
    - Phase 1: generate `<think>...</think>` reasoning.
    - Phase 2: append a deterministic decision prefix and read prompt token logprobs.
  - Supports both single-item and batched scoring.
  - Contains a legacy self-reported-logits path kept mostly for reference, not the main pipeline.

### Backtest

- [backtest/backtester.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py)
  - Batch orchestration for end-to-end evaluation.
  - Calls Agent 1 in batch, then Agent 2 in batch, then fetches realized returns.
- [backtest/dataset_parser.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/backtest/dataset_parser.py)
  - Loads `.csv`, `.parquet`, or Hugging Face datasets.
  - Normalizes columns to `input`, `output`, `answer`, `ticker`.
  - Extracts article text and date ranges from FinGPT-style prompts.
- [backtest/price_fetcher.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/backtest/price_fetcher.py)
  - Fetches daily data from `yfinance`.
  - Computes close-to-close realized returns.
  - Maintains both in-memory and disk caches.
- [backtest/run_backtest.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/backtest/run_backtest.py)
  - CLI wrapper for running backtests and printing metrics.

### Ingestion

- [ingestion/news_fetcher.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/ingestion/news_fetcher.py)
  - Supports `alpaca` and `finnhub`.
  - Finnhub path includes retry, pacing, and deduplication.

## Data Contracts

### `NewsFingerprint`

Fields defined in `agent1/schema.py`:

- Fact extraction fields:
  - `source`
  - `published_at`
  - `headline`
  - `companies_named` (may be empty; ticker fallback applied upstream)
  - `event_keywords` (set to `[event_type.lower()]` for backward compat; optional, default `[]`)
- Sentiment fields:
  - `sentiment_label`: `POSITIVE | NEGATIVE | NEUTRAL`
  - `sentiment_score`: `1.0 | 0.0 | -1.0`
  - `sentiment_confidence`
  - `sentiment_probabilities`
  - `sentiment_logits`
  - `calibration_T`
- Event type fields (new):
  - `event_type`: concrete label or `OTHER`
  - `event_type_confidence`: top softmax probability (or None)
  - `event_type_margin`: top1 − top2 margin (or None)
  - `event_type_method`: `logits_accepted | abstained_low_confidence | abstained_low_margin | event_type_logits_failed`
  - `event_type_logits`: raw log-probabilities keyed by token A-G
  - `event_type_probabilities`: calibrated probabilities keyed by token A-G
  - `secondary_event_type`: second-best concrete label
  - `secondary_event_type_confidence`
- Pass-through context:
  - `article_text`

Validation behavior:

- `event_keywords` are normalized to lowercase.
- `companies_named` validator removed — ticker fallback now applied in `_assemble_fingerprint` before construction.

### `TradingSignal`

Fields defined in `agent2/schema.py`:

- `ticker`
- `direction`: `long | short | neutral`
- `strategy_tag`: always `"event_driven"` (fixed metadata, not a model prediction)
- `confidence`
- `cot`: empty string in no-CoT mode; reasoning text in CoT mode
- Logits audit fields:
  - `signal_logits`: PMI-adjusted logits
  - `raw_signal_logits`: pre-PMI logits
  - `signal_probabilities`
  - `calibration_T`
- Filter fields:
  - `signal_filter_forced_hold`: True when a threshold override applied
  - `signal_filter_reason`: `low_confidence | low_margin | buy_threshold | sell_threshold`

## How Inference Actually Works

### Agent 1 — three-step pipeline

Agent 1 runs three logits-based classifiers per article.

**Step 1: Fact extraction (guided decoding)**
- Prompt: `EXTRACTION_PROMPT` + article text.
- Guided JSON schema enforces output structure.
- Output: source, headline, companies_named, event_keywords.

**Step 2: Sentiment classification**
- Phase 1: CoT generation prompt stops at `</think>`.
- Phase 2: Scoring with `SENTIMENT_DECISION_PREFIX = "Sentiment: "` over tokens `[POSITIVE, NEGATIVE, NEUTRAL]`.
- Python post-processing: `softmax(logits / CALIBRATION_T)` → label, confidence, probabilities.
- Fallback on parse failure: uniform distribution, label = NEUTRAL.

**Step 3: Event type classification**
- Single-phase (no CoT): `EVENT_TYPE_PROMPT` scored directly via `prompt_logprobs`.
- Score tokens: `["A", "B", "C", "D", "E", "F", "G"]` — no OTHER.
- Token mapping (EVENT_TYPE_MAP):
  - A → EARNINGS
  - B → GUIDANCE
  - C → ANALYST_RATING
  - D → LEGAL_REGULATORY
  - E → MNA
  - F → PRODUCT_BUSINESS
  - G → MACRO
- Python post-processing:
  - `softmax(logits / CALIBRATION_T)`
  - `_rank_probabilities` → top_idx, second_idx, top_prob, second_prob, margin
  - `_apply_classification_thresholds`:
    - `top_prob < FINGPT_EVENT_TYPE_MIN_CONFIDENCE` → OTHER, method=abstained_low_confidence
    - `margin < FINGPT_EVENT_TYPE_MIN_MARGIN` → OTHER, method=abstained_low_margin
    - else → EVENT_TYPE_MAP[top_token], method=logits_accepted

### Agent 2 — narrow direction classifier

Agent 2 receives only: ticker, headline, and Agent 1's structured fingerprint.
It does NOT receive the full article text, event keywords, or strategy tag prediction.

**Prompt:** Compact template (`STRATEGY_PROMPT_NO_COT` or `STRATEGY_PROMPT_COT`) with fingerprint fields:
- sentiment_label, sentiment_confidence, sentiment probabilities (p_pos, p_neu, p_neg)
- event_type, event_type_confidence, event_type_margin, event_type_method
- companies_named

**Scoring:** Single-letter tokens `["A", "B", "C"]` via prompt_logprobs.
- A → BUY → long
- B → HOLD → neutral
- C → SELL → short

This avoids tokenization and prior-bias issues from directly scoring `BUY/HOLD/SELL`.

**PMI correction with alpha:**

```
adjusted_logit[t] = raw_logit[t] - FINGPT_PMI_ALPHA * null_logprob[t]
```

- `FINGPT_PMI_ALPHA=1.0` (default) → full PMI correction.
- `FINGPT_PMI_ALPHA=0.0` → no correction (raw logits only).

The null-context prior is:
- Cached in memory for the active process.
- Persisted to `PMI_PRIOR_PATH` on disk.
- Invalidated automatically when model path, decision prefix, or score tokens change.
- NOT invalidated by post-processing changes (pmi_alpha, calibration_T, thresholds).

**Confidence/margin/direction filters** (applied after PMI + softmax):

```
if top_prob < FINGPT_SIGNAL_MIN_CONFIDENCE → force HOLD (low_confidence)
elif margin < FINGPT_SIGNAL_MIN_MARGIN      → force HOLD (low_margin)
elif top==A and prob[A] < FINGPT_BUY_THRESHOLD  → force HOLD (buy_threshold)
elif top==C and prob[C] < FINGPT_SELL_THRESHOLD → force HOLD (sell_threshold)
```

**strategy_tag** is always `"event_driven"` — not predicted by the model.

If Agent 2 cannot parse the scoring result, it returns `None` and writes a diagnostic file.

## Batch Behavior

The repo is optimized around batching.

### Agent 1 batch

`extract_fingerprint_batch(article_texts, tickers=None)` performs:

1. One batched guided-decoding extraction call.
2. One batched CoT + logprobs call for sentiment (2 vLLM calls internally).
3. One batched direct logprobs call for event type (1 vLLM call — no CoT).

`tickers` is an optional list used as companies_named fallback (dataset ticker fallback).

### Agent 2 batch

`generate_signal_batch(fingerprints)` performs:

In **CoT mode** (FINGPT_SIGNAL_USE_COT=True):
1. One batched CoT generation call (stop at `</think>`).
2. One batched A/B/C scoring call.

In **no-CoT mode** (FINGPT_SIGNAL_USE_COT=False, default):
1. One batched A/B/C scoring call (no Phase 1).

### Backtest batch size

- Controlled by `FINGPT_BACKTEST_BATCH_SIZE` env var (default 10).
- Overridable via `--batch-size` CLI argument.

For each batch the practical flow is:

1. Agent 1 fact extraction (guided decoding).
2. Agent 1 sentiment logits.
3. Agent 1 event type logits.
4. Agent 2 direction logits (valid fingerprints only).
5. Price fetch and metric assembly per row.

## Configuration and Environment

Primary config lives in [config.py](/abs/path/C:/Project/FinGPT/FinGPT_Part2/config.py) and `.env`.

### Important environment variables

**Model and engine:**

| Variable | Default | Description |
|---|---|---|
| `FINGPT_MODEL_PATH` | — | Required. Path to local model weights. |
| `FINGPT_ADAPTER_PATH` | — | Present in config but not currently wired into model loading. |
| `SHARE_SINGLE_LLM_BETWEEN_AGENTS` | `false` | Reuse Agent 1 vLLM engine in Agent 2. |
| `FINGPT_CALIBRATION_T` | `1.2` | Softmax temperature for all classifiers. |
| `FINGPT_LOGITS_MAX_TOKENS` | `1024` | CoT token budget for sentiment. |
| `FINGPT_PMI_PRIOR_PATH` | `output/pmi_null_logprobs.json` | Disk cache for Agent 2 PMI prior. |

**Agent 1 — event type classifier:**

| Variable | Default | Description |
|---|---|---|
| `FINGPT_EVENT_TYPE_MIN_CONFIDENCE` | `0.0` | Minimum softmax probability to accept a concrete label; below → OTHER. |
| `FINGPT_EVENT_TYPE_MIN_MARGIN` | `0.0` | Minimum top1−top2 margin; below → OTHER. |

**Agent 2 — PMI and signal filters:**

| Variable | Default | Description |
|---|---|---|
| `FINGPT_PMI_ALPHA` | `1.0` | Scaling factor for PMI correction. 0.0 = no correction. |
| `FINGPT_SIGNAL_MIN_CONFIDENCE` | `0.0` | Force HOLD if top_prob < threshold. |
| `FINGPT_SIGNAL_MIN_MARGIN` | `0.0` | Force HOLD if margin < threshold. |
| `FINGPT_BUY_THRESHOLD` | `0.0` | Force HOLD if top==A and prob[A] < threshold. |
| `FINGPT_SELL_THRESHOLD` | `0.0` | Force HOLD if top==C and prob[C] < threshold. |
| `FINGPT_SIGNAL_USE_COT` | `false` | Enable CoT generation in Agent 2. Default false for backtest speed. |

**Backtest:**

| Variable | Default | Description |
|---|---|---|
| `FINGPT_BACKTEST_STRICT_MODE` | `false` | Skip rows on any logits failure (vs. graceful fallback). |
| `FINGPT_BACKTEST_BATCH_SIZE` | `10` | Articles per vLLM batch. Also overridable via `--batch-size`. |

**Data and diagnostics:**

| Variable | Default | Description |
|---|---|---|
| `FINGPT_YF_CACHE_PATH` | — | Disk cache for realized returns from yfinance. |
| `FINGPT_DIAG_MD_DIR` | `output/diagnostics_md` | Base directory for markdown debug outputs. |
| `NEWS_PROVIDER` | `alpaca` | `alpaca` or `finnhub`. |
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | — | Alpaca news API credentials. |
| `FINNHUB_API_KEY` | — | Finnhub API key. |
| `FINNHUB_TIMEOUT_SEC` | `15` | Finnhub request timeout. |
| `FINNHUB_MAX_CALLS_PER_SEC` | `25` | Finnhub rate limit. |
| `FINNHUB_MAX_RETRIES` | `3` | Finnhub retry count. |
| `FINNHUB_RETRY_BASE_DELAY_SEC` | `0.5` | Finnhub retry base delay. |

### Notable config mismatch

`config.py` still defines Anthropic-related constants:

- `ANTHROPIC_API_KEY`
- `CLAUDE_MODEL`
- `CLAUDE_MAX_TOKENS`
- `CLAUDE_THINKING_BUDGET`

But the active Agent 2 implementation in `agent2/reasoner.py` is local vLLM-based, not Anthropic-based. Those constants currently look like leftovers from an earlier design and should not be treated as part of the main runtime path.

## Runtime Outputs

### Live pipeline output

- JSON file in `output/signals_<timestamp>.json`
- Each record is a serialized `TradingSignal`

### Backtest output

- CSV file, default `output/backtest_results.csv`
- Row-level fields include:
  - dataset metadata
  - Agent 1 sentiment results
  - Agent 2 signal results
  - realized return
  - position
  - strategy return
  - `skipped_reason`

### Common skip reasons

- `fingerprint_failed` — Agent 1 fact extraction or assembly failed.
- `event_type_logits_failed` — Event type logits failed AND `FINGPT_BACKTEST_STRICT_MODE=true`.
- `signal_failed` — Agent 2 signal logits failed or parse error.
- `price_fetch_failed` — Could not fetch realized return from yfinance.

### Diagnostics and logs

- Agent 1 markdown debug outputs:
  - `<diag_dir>/agent1`
- Agent 2 markdown debug outputs:
  - `<diag_dir>/agent2`
- Agent 2 failure diagnostics:
  - `<diag_dir>/agent2_failures`
- Agent 2 CoT logs:
  - `logs/cot_<ticker>_<timestamp>.txt`

## Backtest Metrics

`compute_metrics()` returns:

**Core:**
- `total_rows`, `successful_rows`, `skip_rate`
- `direction_accuracy`, `long_accuracy`, `short_accuracy`
- `long_precision`, `short_precision`
- `mean_strategy_return`, `std_strategy_return`
- `total_pnl`, `gross_return`
- `annualized_sharpe`
- `max_drawdown`
- `vs_fingpt_accuracy`

**Trade counts:**
- `long_trade_count`, `short_trade_count`, `neutral_count`
- `num_trades` (long + short)
- `coverage` (non-neutral rate)
- `abstention_rate`

**Filter diagnostics:**
- `signal_filter_forced_hold_rate`

**Per event type (nested dict, key = event_type label):**
- `event_type_breakdown`: `{event_type: {mean_return, n_rows}}`

Direction labels are compared against realized movement computed by `direction_from_return()`, which uses a default neutral threshold of `0.001`.

## Dependencies

Dependencies are listed in `requirements.txt`. The main runtime groups are:

- Model/runtime:
  - `transformers`
  - `torch`
  - `peft`
  - vLLM is required by the code but is not listed in `requirements.txt`; it must be installed separately in the execution environment.
- Data and evaluation:
  - `pandas`
  - `pyarrow`
  - `datasets`
  - `yfinance`
- Infra:
  - `python-dotenv`
  - `requests`
  - `pydantic`
- Dev:
  - `pytest`
  - `notebook`

## Entry Points Developers Will Actually Use

### Run the live pipeline

```bash
python pipeline.py AAPL MSFT NVDA
```

### Run a full backtest

```bash
python -m backtest.run_backtest --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" --metrics
```

### Run a smaller smoke-test backtest

```bash
python -m backtest.run_backtest --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" --max-rows 50 --metrics
```

### Run a no-CoT backtest with filters

```bash
FINGPT_SIGNAL_USE_COT=false \
FINGPT_EVENT_TYPE_MIN_CONFIDENCE=0.35 \
FINGPT_EVENT_TYPE_MIN_MARGIN=0.05 \
FINGPT_SIGNAL_MIN_CONFIDENCE=0.40 \
FINGPT_SIGNAL_MIN_MARGIN=0.05 \
python -m backtest.run_backtest \
  --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" \
  --max-rows 50 --metrics
```

### Run a CoT debug session

```bash
FINGPT_SIGNAL_USE_COT=true \
python -m backtest.run_backtest \
  --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" \
  --max-rows 10 --metrics
```

### Use a custom local dataset and output path

```bash
python -m backtest.run_backtest --dataset "your_data.csv" --output "output/my_backtest.csv" --metrics
```

### Override batch size

```bash
python -m backtest.run_backtest --dataset "..." --batch-size 20 --metrics
```

## Where To Change Things

### Change extraction behavior

- Prompting: `agent1/prompt.py`
- Parsing/validation: `agent1/extractor.py`, `agent1/schema.py`

### Change sentiment classes or calibration

- Class order: `config.py`
- Decision prefix / scoring prompt: `agent1/prompt.py`
- Softmax behavior: `vllm_logits_client.py` and `agent1/extractor.py`

### Change trading decision behavior

- Strategy prompt and score tokens: `agent2/prompt.py`
- Strategy mapping and PMI logic: `agent2/reasoner.py`
- Signal schema: `agent2/schema.py`

### Change batch size or result assembly

- `backtest/backtester.py`

### Change market data behavior

- `backtest/price_fetcher.py`

### Change live news provider behavior

- `ingestion/news_fetcher.py`

## Current Caveats and Development Notes

### 1. Tests cover deterministic logic only

The tests under `tests/test_extractor.py` and `tests/test_reasoner.py` test pure-Python post-processing:
- `_rank_probabilities`, `_apply_classification_thresholds`, `_process_event_type_result`
- `_process_signal_result` (via module-level patching of config constants)
- EVENT_TYPE_CLASSES/MAP invariants (no OTHER token, correct label mapping)
- Ticker fallback in `_assemble_fingerprint`
- PMI alpha, confidence/margin/direction filters

vLLM is not imported or mocked in these tests — they run without GPU.

### 2. README and comments may still reflect transitional design history

Some repo text still references older assumptions or intermediate states. When in doubt, treat the runtime code in:

- `agent1/extractor.py`
- `agent2/reasoner.py`
- `vllm_logits_client.py`
- `backtest/backtester.py`

as the source of truth.

### 3. `FINGPT_ADAPTER_PATH` is not currently used in model loading

It is exposed in config and `.env.example`, but the current `_load_model()` implementations do not apply a LoRA adapter path.

### 4. vLLM is a hard runtime dependency

The main code imports vLLM dynamically, but the environment must provide it. This is especially important when recreating the project outside the notebook environment.

## Recommended Development Workflow

1. Start from `config.py`, `.env.example`, and the relevant entry point.
2. For inference changes, trace:
   - prompt file
   - schema file
   - extractor/reasoner
   - `vllm_logits_client.py`
3. For evaluation changes, trace:
   - `dataset_parser.py`
   - `backtester.py`
   - `price_fetcher.py`
4. Treat tests as needing modernization before using them as a safety net.
5. Use the notebooks mainly for experiments and reruns, not as the primary source of architecture truth.
