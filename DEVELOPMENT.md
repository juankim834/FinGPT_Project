# FinGPT Part 2 Development Guide

This document describes the repository as it exists now. It is aligned to the current code in the repo and the latest generated artifacts under `output/`, including the dashboard files and recent backtest/grid-search summaries.

## Purpose

This repo implements a local, inference-only financial news pipeline with two agents:

1. `Agent 1` converts raw news text into a structured `NewsFingerprint`.
2. `Agent 2` converts that fingerprint into a trading direction.
3. The `backtest` package runs the same logic on historical data and evaluates realized returns.

The main design choice is unchanged: decisions come from real token log-probabilities read from vLLM `prompt_logprobs`, then deterministic Python post-processing applies calibration, PMI correction, and filtering.

## Current Repository Shape

- [pipeline.py](/C:/Project/FinGPT/FinGPT_Part2/pipeline.py): live fetch-to-signal entry point
- [agent1/extractor.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/extractor.py): fact extraction, sentiment scoring, event-type scoring
- [agent1/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/schema.py): `NewsFingerprint`
- [agent2/reasoner.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/reasoner.py): signal scoring, PMI prior handling, signal filtering
- [agent2/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/schema.py): `TradingSignal`
- [backtest/backtester.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py): end-to-end backtest, repricing, metrics
- [backtest/pmi_grid_search.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/pmi_grid_search.py): offline hyperparameter sweeps over `pmi_alpha` and confidence thresholds
- [backtest/dataset_parser.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/dataset_parser.py): dataset normalization and row building
- [ingestion/news_fetcher.py](/C:/Project/FinGPT/FinGPT_Part2/ingestion/news_fetcher.py): Alpaca / Finnhub ingestion

## End-To-End Workflow

There are two main execution modes.

### Live / fetch mode

Entry point: [pipeline.py](/C:/Project/FinGPT/FinGPT_Part2/pipeline.py)

Flow:

1. `run_pipeline(tickers, limit, as_of_timestamp)` is called.
2. `fetch_recent_articles(...)` pulls candidate articles from Alpaca or Finnhub.
3. The fetch layer applies leakage-safe filtering:
   - keep only `created_at <= as_of_timestamp`
   - drop articles with missing or unparseable timestamps
   - keep only the latest eligible article per requested ticker
4. For each retained article:
   - `article_text = headline + summary`
   - `extract_fingerprint(article_text, ticker=..., headline=...)`
   - `generate_signal(fingerprint)`
5. Valid signals are written to `output/signals_<timestamp>.json`

Important live-data rule:

- `ticker` and `headline` are upstream metadata, not re-inferred by Agent 2.
- Agent 2 does not re-read the full article body outside the fingerprint contract.

### Backtest mode

Entry point: [backtest/run_backtest.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/run_backtest.py) and [backtest/backtester.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py)

Flow:

1. Load a Hugging Face dataset ID or a local CSV/parquet file.
2. Normalize the source columns to `input`, `output`, `answer`, `ticker`.
3. Build backtest rows with:
   - `ticker`
   - `headline`
   - `article_text`
   - `start_date`
   - `end_date`
   - `fingpt_label`
4. Process rows in batches:
   - Agent 1 batch extraction
   - Agent 2 batch signal generation
   - realized return fetch from yfinance
   - row flattening to CSV
5. Write the detailed backtest CSV

Important backtest rule:

- For multi-news Hugging Face prompts, the parser uses the first headline as the external `headline`.
- Dataset `ticker` is passed directly to Agent 1 as the authoritative ticker.

## Data Contracts

### `NewsFingerprint`

Defined in [agent1/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/schema.py).

Key fields:

- metadata: `ticker`, `headline`, `source`, `published_at`
- sentiment: `sentiment_label`, `sentiment_confidence`, `sentiment_probabilities`, `sentiment_logits`
- event type: `event_type`, `event_type_confidence`, `event_type_margin`, `event_type_method`
- pass-through: `article_text`

Current behavior:

- `ticker` is no longer inferred from `companies_named[0]`
- `headline` is preserved from upstream inputs when available
- `source` and `published_at` are normalized to strings
- `companies_named` can be backfilled from `ticker`
- `event_keywords` is compatibility-only and is derived from the final event label

### `TradingSignal`

Defined in [agent2/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/schema.py).

Key fields:

- `direction`
- `confidence`
- `strategy_tag`
- `raw_signal_logits`
- `pmi_null_logprobs`
- `signal_logits`
- `signal_probabilities`
- `signal_filter_forced_hold`
- `signal_filter_reason`

Current behavior:

- `strategy_tag` is fixed to `"event_driven"`
- `cot` is empty when `FINGPT_SIGNAL_USE_COT=False`

## Every LLM Call

### Agent 1 call 1: guided fact extraction

Code path:

- [agent1/extractor.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/extractor.py)

Prompt source:

- [agent1/prompt.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/prompt.py): `EXTRACTION_PROMPT`

Input:

- raw `article_text`

Requested fields:

- `source`
- `published_at`
- `companies_named`
- `event_keywords`

Mechanism:

1. Build the extraction prompt.
2. Request schema-guided JSON generation from vLLM.
3. Parse as JSON, then balanced JSON fallback, then markdown-style fallback.

Current degradation path:

- If extraction parsing fails, the row does not die immediately.
- The code falls back to an empty extraction payload so sentiment and event-type scoring can still proceed.

### Agent 1 call 2: sentiment scoring

Code path:

- [agent1/extractor.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/extractor.py): `_score_sentiment`, `_process_sentiment_result`

Prompt source:

- [agent1/prompt.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/prompt.py): `SENTIMENT_PROMPT`

Label set:

- `POSITIVE`
- `NEGATIVE`
- `NEUTRAL`

Mechanism:

1. Build a direct classification prompt.
2. Read next-token log-probabilities with:
   - `decision_prefix = "Sentiment: "`
   - `use_cot = False`
3. Convert logits to calibrated probabilities with `softmax(logits / CALIBRATION_T)`.
4. Produce label, confidence, probability dict, and raw logits.

Fallback:

- If logits processing fails, the row falls back to uniform probabilities, zero logits, and `NEUTRAL`.

### Agent 1 call 3: event-type scoring

Code path:

- [agent1/extractor.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/extractor.py): `_score_event_type`, `_process_event_type_result`

Prompt source:

- [agent1/prompt.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/prompt.py): `EVENT_TYPE_PROMPT`

Score tokens:

- `A`, `B`, `C`, `D`, `E`, `F`, `G`

Mapping:

- `A -> EARNINGS`
- `B -> GUIDANCE`
- `C -> ANALYST_RATING`
- `D -> LEGAL_REGULATORY`
- `E -> MNA`
- `F -> PRODUCT_BUSINESS`
- `G -> MACRO`

Mechanism:

1. Score A-G token log-probabilities.
2. Convert to probabilities with calibration.
3. Rank top-1 and top-2.
4. Apply Python-side abstention rules:
   - below `FINGPT_EVENT_TYPE_MIN_CONFIDENCE` -> `OTHER`
   - below `FINGPT_EVENT_TYPE_MIN_MARGIN` -> `OTHER`

Important design point:

- `OTHER` is never model-scored. It is assigned only by post-processing.

### Agent 2 call 1: optional PMI null-context prior

Code path:

- [agent2/reasoner.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/reasoner.py): `_compute_null_logprobs`

Purpose:

- Compute the model prior over the three strategy score tokens without real article context.

Mechanism:

1. Build a synthetic neutral fingerprint.
2. Score A/B/C once.
3. Cache the result in memory and optionally on disk via `PMI_PRIOR_PATH`.

This is per model/config combination, not per article.

### Agent 2 call 2: strategy scoring

Code path:

- [agent2/reasoner.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/reasoner.py)

Prompt source:

- [agent2/prompt.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/prompt.py)

Input actually visible to Agent 2:

- `ticker`
- `headline`
- Agent 1 sentiment outputs
- Agent 1 event-type outputs
- `companies_named`

Notably excluded:

- full raw article body as a fresh read
- raw event keyword extraction

Score tokens:

- `A`, `B`, `C`

Mapping:

- `A -> BUY -> long`
- `B -> HOLD -> neutral`
- `C -> SELL -> short`

Mechanism:

1. Build the compact fingerprint prompt.
2. Optionally generate CoT if `FINGPT_SIGNAL_USE_COT=True`.
3. Score A/B/C.
4. Optionally apply PMI correction:
   - `adjusted = raw - pmi_alpha * null`
5. Convert to calibrated probabilities.
6. Apply signal filters:
   - low confidence -> HOLD
   - low margin -> HOLD
   - weak BUY -> HOLD
   - weak SELL -> HOLD

## Batch Behavior

### Agent 1 batch behavior

`extract_fingerprint_batch(...)` performs:

1. one batched guided extraction call
2. one batched sentiment scoring call
3. one batched event-type scoring call
4. per-item fingerprint assembly in Python

### Agent 2 batch behavior

`generate_signal_batch(...)` performs:

- no-CoT mode: one batched A/B/C scoring call
- CoT mode: one batched CoT generation call, then one batched A/B/C scoring call

### Backtest batch behavior

Per batch:

1. run Agent 1 batch extraction
2. keep valid fingerprints
3. run Agent 2 batch scoring
4. fetch realized returns
5. flatten results into CSV rows

## Failure and Degradation Modes

Current behavior:

- malformed fact extraction -> empty extraction payload fallback
- sentiment failure -> `NEUTRAL` with uniform probabilities
- event-type failure -> `OTHER`, method marked as `event_type_logits_failed`
- Agent 2 scoring failure -> row marked `signal_failed`
- price fetch failure -> row kept with `price_fetch_failed`, but excluded from successful evaluation rows

## Configuration That Matters Most

Main file: [config.py](/C:/Project/FinGPT/FinGPT_Part2/config.py)

Important current defaults:

- `CALIBRATION_T = 1.2`
- `FINGPT_PMI_ALPHA = 1.0`
- `FINGPT_SIGNAL_MIN_CONFIDENCE = 0.0`
- `FINGPT_SIGNAL_MIN_MARGIN = 0.0`
- `FINGPT_SIGNAL_USE_COT = False`
- `FINGPT_BACKTEST_STRICT_MODE = False`
- `FINGPT_BACKTEST_BATCH_SIZE = 10`

This means the default baseline is:

- calibrated but permissive
- PMI-corrected by default
- no Agent 2 CoT in backtests
- intended to be tuned further via offline sweeps instead of hard-coded thresholds

## Dashboard Artifacts

The repo now includes static dashboard outputs under [output/dashboard](/C:/Project/FinGPT/FinGPT_Part2/output/dashboard).

### `pmi_alpha_grid_search_backtest.html`

File: [pmi_alpha_grid_search_backtest.html](/C:/Project/FinGPT/FinGPT_Part2/output/dashboard/pmi_alpha_grid_search_backtest.html)

This dashboard lets you inspect six `pmi_alpha` settings:

- `0.00`
- `0.25`
- `0.50`
- `0.75`
- `1.00`
- `1.25`

It contains:

- alpha selector buttons
- KPI cards for total return, mean return per trade, hit rate, Sharpe, Sortino, and max drawdown
- position breakdown bar for long / neutral / short counts
- cumulative-return equity curve for the selected alpha
- summary comparison chart across all alphas

### `alpha_confidence_2d_grid_search.html`

File: [alpha_confidence_2d_grid_search.html](/C:/Project/FinGPT/FinGPT_Part2/output/dashboard/alpha_confidence_2d_grid_search.html)

This dashboard visualizes a 2D sweep over `pmi_alpha` and `signal_min_confidence`.

It contains:

- heatmaps for:
  - total PnL
  - Sharpe
  - direction accuracy
  - trade count
- a detail panel for any selected cell
- a line chart of PnL by confidence threshold for each alpha
- a trade-count / forced-hold summary panel by confidence threshold

This dashboard is backed by [output/alpha_confidence_grid_search.csv](/C:/Project/FinGPT/FinGPT_Part2/output/alpha_confidence_grid_search.csv).

### `preview.html`

File: [preview.html](/C:/Project/FinGPT/FinGPT_Part2/output/dashboard/preview.html)

This is the current long-only grid-search dashboard. It uses a margin-filtered BUY rule rather than the symmetric long / neutral / short post-processing used in the main backtest dashboards.

The decision rule is:

```python
adjusted_logit = raw_logit - alpha * null_logit
prob = softmax(adjusted_logit)

buy_margin = prob_buy - max(prob_hold, prob_sell)

if prob_buy >= confidence_threshold and buy_margin >= margin_threshold:
    position = 1
else:
    position = 0
```

Interpretation:

- long-only: there is no short position
- BUY that passes both thresholds becomes a long
- HOLD and SELL both become cash / no position
- the extra `buy_margin` term makes BUY compete explicitly against the stronger of HOLD and SELL

The dashboard includes:

- best gross combination
- best net 10bp combination
- top 20 combinations by gross PnL
- full grid table
- per-cell metrics:
  - `alpha`
  - `confidence`
  - `longs`
  - `coverage`
  - `long precision > 10bp`
  - `gross`
  - `net10bp`
  - `net20bp`
  - `Sharpe`
  - `Sortino`
  - `MaxDD`

## Current Backtest Performance Snapshot

These observations come from the current generated artifacts:

- [output/pmi_alpha_grid_search.csv](/C:/Project/FinGPT/FinGPT_Part2/output/pmi_alpha_grid_search.csv)
- [output/alpha_confidence_grid_search.csv](/C:/Project/FinGPT/FinGPT_Part2/output/alpha_confidence_grid_search.csv)
- [output/backtest_20260506T051302Z_repriced.csv](/C:/Project/FinGPT/FinGPT_Part2/output/backtest_20260506T051302Z_repriced.csv)
- the two dashboard HTML files above

### Coverage

For the current run family:

- `300` raw rows were processed
- `290` rows were successfully priced and scored in the summary artifacts
- `10` rows were excluded because of `price_fetch_failed`

### One-dimensional `pmi_alpha` sweep

The current outputs show that performance is highly sensitive to `pmi_alpha`, and the repo default `pmi_alpha=1.0` is not the best setting on this sample.

Best-performing alphas by total PnL:

- `alpha=0.0`: `total_pnl=0.2814`, `annualized_sharpe=0.2870`, `direction_accuracy=44.14%`
- `alpha=0.5`: `total_pnl=0.1373`
- `alpha=0.25`: `total_pnl=0.1326`

Weaker-performing alphas:

- `alpha=0.75`: `total_pnl=-0.3657`
- `alpha=1.0`: `total_pnl=-0.2632`
- `alpha=1.25`: `total_pnl=-0.3162`

The failure mode is visible in the position mix:

- at `alpha=1.0`, the successful rows are dominated by shorts: `249 short` vs `12 long`
- at `alpha=1.25`, that bias gets even stronger: `270 short` vs `7 long`

So on the current backtest sample, stronger PMI correction appears to over-push the model toward the sell side.

### Two-dimensional `alpha + confidence` sweep

The best combination in the current 2D sweep is:

- `pmi_alpha=0.0`
- `signal_min_confidence=0.35`

Metrics for that cell:

- `total_pnl=0.3028`
- `annualized_sharpe=0.3101`
- `direction_accuracy=43.79%`
- `num_trades=232`
- `forced_hold_rate=1.33%`

This slightly improves on the looser `alpha=0.0, confidence=0.30` setting and clearly outperforms the default-style `alpha=1.0, confidence=0.0` baseline present in the current config.

Broad trend from the sweep:

- modest confidence filtering can improve PnL and Sharpe
- aggressive confidence thresholds quickly raise forced holds
- once confidence reaches roughly `0.45+`, trade count falls sharply for many alpha settings
- many high-threshold combinations become low-coverage, high-abstention configurations

### Long-only performance from `preview.html`

The current long-only dashboard shows a stronger top-line result than the default symmetric direction backtest, but it is evaluating a different decision rule: only high-conviction BUY signals become trades, while everything else stays in cash.

The best gross configuration currently shown is:

- `alpha=0.05`
- `confidence=0.30`

Metrics for that setting:

- `gross=0.4306`
- `Sharpe=0.495`
- `MaxDD=0.3798`
- `longs=165`
- `coverage=56.9%`
- `long precision > 10bp = 58.8%`

That same configuration is also the dashboard's best net 10bp result:

- `net10bp=0.2656`
- `net20bp=0.1006`

Broad trend from the current long-only grid:

- smaller `alpha` values, especially around `0.00` to `0.25`, are currently the strongest region
- `confidence=0.30` to `0.35` is the best-performing range in the present artifact
- once confidence is pushed to `0.45+`, coverage falls quickly
- higher thresholds may improve selectivity, but often at the cost of too few trades to preserve total return

## Known Caveat

One current caveat showed up while validating the generated artifacts:

- when some result CSVs are read back from disk, blank `skipped_reason` values may not normalize the same way as in-memory results
- that can cause `compute_metrics(...)` on a reloaded CSV to undercount or even exclude successful rows

The dashboard and grid-search conclusions above remain valid because they come from the already-generated summary CSVs and dashboards, not from that problematic re-read path.

If you are improving the repo, this `skipped_reason` normalization issue is a good next cleanup target.

## Files You Will Usually Edit

If you want to change extraction behavior:

- [agent1/prompt.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/prompt.py)
- [agent1/extractor.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/extractor.py)
- [agent1/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/schema.py)

If you want to change strategy behavior:

- [agent2/prompt.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/prompt.py)
- [agent2/reasoner.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/reasoner.py)
- [agent2/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/schema.py)

If you want to change backtesting and offline search:

- [backtest/backtester.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py)
- [backtest/pmi_grid_search.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/pmi_grid_search.py)
- [backtest/dataset_parser.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/dataset_parser.py)
- [backtest/price_fetcher.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/price_fetcher.py)

If you want to change live ingestion:

- [ingestion/news_fetcher.py](/C:/Project/FinGPT/FinGPT_Part2/ingestion/news_fetcher.py)
- [pipeline.py](/C:/Project/FinGPT/FinGPT_Part2/pipeline.py)
