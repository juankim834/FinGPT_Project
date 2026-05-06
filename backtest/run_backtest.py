"""
CLI entrypoint for running news-to-signal backtests.
"""

from __future__ import annotations

import argparse

from backtest.backtester import compute_metrics, reprice_backtest_results, run_backtest


def _print_metrics_table(metrics: dict) -> None:
    # Print scalar metrics first, then nested event_type_breakdown separately.
    scalar_items = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    nested_items = {k: v for k, v in metrics.items() if isinstance(v, dict)}

    keys = list(scalar_items.keys())
    key_width = max(len(key) for key in keys) + 2
    print("=" * (key_width + 24))
    print(f"{'Metric':<{key_width}}Value")
    print("=" * (key_width + 24))
    for key in keys:
        value = scalar_items[key]
        if isinstance(value, float):
            print(f"{key:<{key_width}}{value:.6f}")
        else:
            print(f"{key:<{key_width}}{value}")
    print("=" * (key_width + 24))

    for section_name, section in nested_items.items():
        print(f"\n{section_name}:")
        print("-" * 40)
        for sub_key, sub_val in section.items():
            if isinstance(sub_val, dict):
                parts = ", ".join(f"{k}={v}" for k, v in sub_val.items())
                print(f"  {sub_key}: {parts}")
            else:
                print(f"  {sub_key}: {sub_val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinGPT news-to-signal backtest.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to parquet/csv dataset file.",
    )
    parser.add_argument(
        "--output",
        default="output/backtest_results.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max rows to process for quick tests.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print backtest metrics summary after completion.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of articles per vLLM batch. Overrides FINGPT_BACKTEST_BATCH_SIZE.",
    )
    parser.add_argument(
        "--resume-csv",
        default=None,
        help="Existing backtest CSV to re-price and re-score metrics without rerunning Agent 1/2.",
    )
    parser.add_argument(
        "--no-refresh-prices",
        action="store_true",
        help="Reuse cached yfinance results when repricing an existing CSV.",
    )
    args = parser.parse_args()

    if args.resume_csv:
        results = reprice_backtest_results(
            args.resume_csv,
            output_path=args.output,
            refresh_prices=not args.no_refresh_prices,
        )
    else:
        if not args.dataset:
            parser.error("--dataset is required unless --resume-csv is provided.")
        results = run_backtest(
            dataset_path=args.dataset,
            output_path=args.output,
            max_rows=args.max_rows,
            batch_size=args.batch_size,
        )
    if args.metrics:
        metrics = compute_metrics(results)
        _print_metrics_table(metrics)


if __name__ == "__main__":
    main()
