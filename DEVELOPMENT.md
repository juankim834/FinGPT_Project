# FinGPT Two-Agent Signal Pipeline — Development Guide

## Current State

- Inference runs on local vLLM using DeepSeek-R1-Distill-Llama-8B.
- Agent 1 extracts `NewsFingerprint` + sentiment.
- Agent 2 converts fingerprint into `TradingSignal`.
- Backtest mode is first-class and uses:
  - Hugging Face dataset loading (`FinGPT/fingpt-forecaster-dow30-202305-202405`)
  - yfinance realized returns
  - strict skip-on-failure behavior
- Colab notebook is backtest-only after setup cells.

---

## 1) Architecture (Current)

### Backtest Path (Primary)

```
HF dataset / local parquet-csv
  -> backtest/dataset_parser.py
  -> agent1/extractor.py
  -> agent2/reasoner.py
  -> backtest/price_fetcher.py
  -> backtest/backtester.py
  -> output/backtest_results*.csv + metrics
```

### Live News Path (Optional)

```
ingestion/news_fetcher.py
  -> agent1/extractor.py
  -> agent2/reasoner.py
  -> pipeline.py
```

---

## 2) Key Files

```
FinGPT_Part2/
├── config.py
├── pipeline.py
├── agent1/
│   ├── extractor.py
│   ├── prompt.py
│   └── schema.py
├── agent2/
│   ├── reasoner.py
│   ├── prompt.py
│   └── schema.py
├── backtest/
│   ├── __init__.py
│   ├── dataset_parser.py
│   ├── price_fetcher.py
│   ├── backtester.py
│   └── run_backtest.py
├── ingestion/news_fetcher.py
├── notebooks/demo.ipynb
└── DEVELOPMENT.md
```

---

## 3) Model and Prompt Constraints

- Use vLLM structured decoding via `StructuredOutputParams`.
- Do not use `GuidedDecodingParams`.
- Prompts are user-turn only (no system-role dependency).
- Prompts use direct task instructions (no persona framing).
- Agent 1 and Agent 2 prompts include compact "good output" JSON examples.

---

## 4) Agent Behavior

### Agent 1 (`agent1/extractor.py`)

- `extract_fingerprint(article_text) -> Optional[NewsFingerprint]`
- Structured extraction + separate sentiment classification.
- Uses shared/injected vLLM engine when available.
- Loader now reuses injected engine even if tokenizer is not yet ready.

### Agent 2 (`agent2/reasoner.py`)

- `generate_signal(fingerprint) -> Optional[TradingSignal]`
- Structured output expected and validated by `TradingSignal`.
- Strict mode: if structured parse fails, returns `None` (no permissive fallback signal).
- Saves raw diagnostic markdown output for each generation attempt.

---

## 5) Backtest Data Parsing Notes

`backtest/dataset_parser.py` supports:

- Local `.parquet`
- Local `.csv`
- Hugging Face dataset IDs

HF split selection priority:

1. `test`
2. `validation`
3. `train`
4. first available split

Multi-headline prompt handling:

- Extracts `[Headline]` + `[Summary]` pairs.
- Preserves structure as numbered news blocks in `article_text`.

Duplicate column safety:

- Handles duplicate labels / `Series` values from HF->pandas conversion.
- Avoids ambiguous truth-value errors in row parsing.

---

## 6) Price Fetching and Caching

`backtest/price_fetcher.py` caches realized returns:

- In-memory cache for current process.
- Persistent disk cache JSON for cross-run reuse.

Cache path:

- Env var `FINGPT_YF_CACHE_PATH`, default `output/yfinance_return_cache.json`

Useful helper:

- `get_cache_stats()` returns memory and disk cache counts.

---

## 7) Colab Reload and Runtime Hygiene

After changing local project Python files in Colab Drive mounts, reload modules:

```python
import importlib
import agent1.prompt as p1, agent1.extractor as e1
import agent2.prompt as p2, agent2.reasoner as r2
import backtest.dataset_parser as dp
import backtest.price_fetcher as pf
import backtest.backtester as bt

for m in [p1, e1, p2, r2, dp, pf, bt]:
    importlib.reload(m)
```

If behavior still looks stale:

- Restart runtime and rerun from setup cells.

---

## 8) Troubleshooting Playbook

### A) `ValueError: Unsupported dataset format. Use .parquet or .csv`

Cause:

- Old `dataset_parser` module still loaded in notebook kernel.

Fix:

1. Reload `backtest.dataset_parser`.
2. Verify `print(dp.__file__)` points to intended repo path.

### B) `ValueError: The truth value of a Series is ambiguous`

Cause:

- HF->pandas row value is `Series` (duplicate labels), not scalar.

Fix:

- Use current parser code that normalizes duplicate columns and safe row extraction.

### C) Agent 2 outputs free-form reasoning, not JSON

Symptoms:

- Diagnostic `.md` starts with natural language reasoning.

Cause:

- Structured generation not enforced at runtime for that call, or model drift.

Current behavior:

- Agent 2 strict mode now fails row as `signal_failed` instead of manufacturing fallback signal.

Actions:

1. Confirm `StructuredOutputParams` path is active.
2. Reload `agent2.reasoner` and `agent2.prompt`.
3. Increase `max_tokens` if truncation suspected.
4. Inspect diagnostics in `FINGPT_DIAG_MD_DIR/agent2`.

### D) vLLM engine init crash (`Engine core initialization failed`)

Typical trigger:

- Agent attempts to spawn a second engine in constrained Colab/GPU state.

Current mitigation:

- Both agents reuse injected shared engine if present.
- Loader returns early when engine exists.

Actions:

1. Ensure shared engine injection cell was run.
2. Reload agent modules.
3. Restart runtime if a stale crashed engine process remains.

---

## 9) Operational Recommendations

- Keep backtest notebook on test split by default.
- Keep strict parse behavior in Agent 2 to avoid hidden bad generations.
- Keep diagnostics markdown enabled in non-production runs.
- Track skip reasons (`fingerprint_failed`, `signal_failed`, `price_fetch_failed`) as health signals.

---

## 10) Quick Commands

Run backtest from CLI:

```bash
python -m backtest.run_backtest --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" --metrics
```

Run subset:

```bash
python -m backtest.run_backtest --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" --max-rows 50 --metrics
```
