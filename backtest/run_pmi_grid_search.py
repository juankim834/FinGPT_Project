"""
CLI entrypoint for offline PMI-alpha grid search.
"""

from __future__ import annotations

import argparse

from backtest.pmi_grid_search import parse_alpha_grid, run_pmi_alpha_grid_search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline PMI-alpha grid search on an existing backtest CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Existing backtest CSV with raw_signal_logprob_* and realized_return columns.",
    )
    parser.add_argument(
        "--alphas",
        default="1.0",
        help='Comma-separated alpha list, e.g. "0,0.25,0.5,0.75,1.0,1.25".',
    )
    parser.add_argument(
        "--output",
        default="output/pmi_alpha_grid_search.csv",
        help="Where to save the summary CSV.",
    )
    parser.add_argument(
        "--detailed-output-dir",
        default=None,
        help="Optional directory for per-alpha detailed backtest CSVs.",
    )
    args = parser.parse_args()

    summary = run_pmi_alpha_grid_search(
        args.input,
        alphas=parse_alpha_grid(args.alphas),
        output_path=args.output,
        detailed_output_dir=args.detailed_output_dir,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
