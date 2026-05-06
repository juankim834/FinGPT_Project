"""
CLI entrypoint for Alpaca-news-driven backtests.
"""

from __future__ import annotations

import argparse

from backtest.alpaca_news_pipeline import run_alpaca_backtest_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Alpaca news by config, build a backtest dataset, and optionally run the existing backtester."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the Alpaca backtest JSON config file.",
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only fetch/cache/build the dataset; skip the existing backtest stage.",
    )
    args = parser.parse_args()

    result = run_alpaca_backtest_pipeline(
        args.config,
        run_existing_backtest=not args.dataset_only,
    )

    print(f"dataset_path={result['dataset_path']}")
    print(f"article_count={result['article_count']}")
    print(f"dataset_rows={result['dataset_rows']}")
    if not args.dataset_only:
        print(f"backtest_output_path={result['backtest_output_path']}")
        print(f"backtest_rows={result.get('backtest_rows', 0)}")


if __name__ == "__main__":
    main()
