"""
Dataset parsing utilities for FinGPT backtesting.
"""

from __future__ import annotations

import re

import pandas as pd

_INST_TAG_RE = re.compile(r"\[/?INST\]|<<SYS>>|<</SYS>>", re.IGNORECASE)
_DATE_RANGE_RE = re.compile(
    r"From\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from local parquet/CSV or Hugging Face dataset ID.
    Returns a DataFrame with normalized columns: input, output, answer, ticker.
    """
    lower = path.lower()
    if lower.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif lower.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        try:
            from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset
        except Exception as exc:
            raise ValueError(
                "Unsupported dataset source. Use .parquet/.csv or install `datasets` "
                "to load a Hugging Face dataset ID."
            ) from exc

        loaded = hf_load_dataset(path)
        if isinstance(loaded, Dataset):
            hf_ds = loaded
        elif isinstance(loaded, DatasetDict):
            split_priority = ("test", "validation", "train")
            split = next((name for name in split_priority if name in loaded), None)
            if split is None:
                split = next(iter(loaded.keys()))
            hf_ds = loaded[split]
        else:
            raise ValueError(f"Unexpected dataset object type: {type(loaded).__name__}")
        df = hf_ds.to_pandas()

    # Normalize common naming variants to required names.
    normalized_columns = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=normalized_columns)
    column_aliases = {
        "input_text": "input",
        "prompt": "input",
        "response": "output",
        "label": "answer",
        "symbol": "ticker",
    }
    df = df.rename(columns=column_aliases)

    required = ["input", "output", "answer", "ticker"]
    optional = ["skip_llm", "forced_signal", "skip_reason", "pass_reason"]
    normalized = pd.DataFrame(index=df.index)
    for col in required:
        if col not in df.columns:
            normalized[col] = ""
            continue

        selected = df[col]
        # Handle duplicate column labels that return a DataFrame.
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[:, 0]
        normalized[col] = selected

    for col in optional:
        if col not in df.columns:
            normalized[col] = ""
            continue
        selected = df[col]
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[:, 0]
        normalized[col] = selected

    return normalized.reset_index(drop=True)


def extract_article_text(input_prompt: str) -> str:
    """
    Extract headline/news section from raw [INST] prompt content.
    """
    text = input_prompt or ""
    text = _INST_TAG_RE.sub(" ", text)
    text = text.strip()

    ticker_match = re.search(r"(^|\n)\s*Ticker:\s*([^\n]+)", text, re.IGNORECASE)
    ticker_line = ""
    if ticker_match:
        ticker_value = re.sub(r"\s+", " ", ticker_match.group(2)).strip()
        if ticker_value:
            ticker_line = f"Ticker: {ticker_value}"

    start_idx = text.find("[Headline]:")
    if start_idx < 0:
        return ""

    end_idx = text.find("[Basic Financials]:", start_idx)
    if end_idx < 0:
        end_idx = len(text)

    news_block = text[start_idx:end_idx]

    # Keep headline/summary pairs explicit so multi-news prompts remain structured.
    pattern = re.compile(
        r"\[Headline\]:\s*(.*?)\s*\[Summary\]:\s*(.*?)(?=\s*\[Headline\]:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    pairs = pattern.findall(news_block)
    if pairs:
        rendered: list[str] = []
        for idx, (headline, summary) in enumerate(pairs, start=1):
            h = re.sub(r"\s+", " ", headline).strip()
            s = re.sub(r"\s+", " ", summary).strip()
            if not h and not s:
                continue
            rendered.append(f"News {idx}\nHeadline: {h}\nSummary: {s}")
        body = "\n\n".join(rendered).strip()
        return f"{ticker_line}\n\n{body}".strip() if ticker_line else body

    # Fallback for prompts that don't strictly follow headline/summary pair format.
    news_block = re.sub(r"\[(Headline|Summary)\]:", " ", news_block, flags=re.IGNORECASE)
    news_block = re.sub(r"\s+", " ", news_block).strip()
    return f"{ticker_line}\n\n{news_block}".strip() if ticker_line else news_block


def extract_primary_headline(input_prompt: str) -> str:
    """
    Extract the first headline from a raw [INST] prompt.

    For multi-news prompts without per-item timestamps, the first headline is
    treated as the primary / most relevant headline for downstream use.
    """
    text = input_prompt or ""
    text = _INST_TAG_RE.sub(" ", text).strip()

    match = re.search(r"\[Headline\]:\s*(.*?)(?=\s*\[Summary\]:|\s*\[Headline\]:|\Z)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def extract_date_range(input_prompt: str) -> tuple[str, str]:
    """
    Parse `From YYYY-MM-DD to YYYY-MM-DD` from prompt text.
    """
    match = _DATE_RANGE_RE.search(input_prompt or "")
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def parse_fingpt_label(answer: str) -> str:
    """
    Normalize FinGPT answer label to up/down/neutral.
    """
    normalized = (answer or "").strip().lower()
    if not normalized:
        return "no_label_provided"
    if "up" in normalized:
        return "up"
    if "down" in normalized:
        return "down"
    return "neutral"


def build_backtest_rows(df: pd.DataFrame) -> list[dict]:
    """
    Convert dataset rows into normalized backtest rows.
    """
    def _row_value(row_obj: pd.Series, key: str) -> str:
        raw = row_obj.get(key, "")
        if isinstance(raw, pd.Series):
            for value in raw.tolist():
                if pd.notna(value):
                    return str(value)
            return ""
        if pd.isna(raw):
            return ""
        return str(raw)

    def _row_bool(row_obj: pd.Series, key: str) -> bool:
        raw = row_obj.get(key, False)
        if isinstance(raw, pd.Series):
            raw = raw.iloc[0] if len(raw) else False
        if pd.isna(raw):
            return False
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    rows: list[dict] = []
    for _, row in df.iterrows():
        raw_input = _row_value(row, "input")
        ticker = _row_value(row, "ticker").strip().upper()
        skip_llm = _row_bool(row, "skip_llm")
        forced_signal = _row_value(row, "forced_signal").strip()
        skip_reason = _row_value(row, "skip_reason").strip()
        pass_reason = _row_value(row, "pass_reason").strip()

        if skip_llm:
            rows.append(
                {
                    "ticker": ticker,
                    "headline": "",
                    "start_date": "",
                    "end_date": "",
                    "article_text": "",
                    "fingpt_label": "neutral",
                    "skip_llm": True,
                    "forced_signal": forced_signal,
                    "skip_reason": skip_reason,
                    "pass_reason": pass_reason,
                }
            )
            continue

        article_text = extract_article_text(raw_input)
        if not ticker or not article_text:
            continue

        start_date, end_date = extract_date_range(raw_input)
        fingpt_label = parse_fingpt_label(_row_value(row, "answer"))

        rows.append(
            {
                "ticker": ticker,
                "headline": extract_primary_headline(raw_input),
                "start_date": start_date,
                "end_date": end_date,
                "article_text": article_text,
                "fingpt_label": fingpt_label,
                "skip_llm": False,
                "forced_signal": forced_signal,
                "skip_reason": skip_reason,
                "pass_reason": pass_reason,
            }
        )
    return rows

