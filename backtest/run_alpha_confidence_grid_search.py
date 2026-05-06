"""
CLI entrypoint for offline 2D grid search over PMI alpha and confidence threshold.
"""

from __future__ import annotations

import argparse

from backtest.pmi_grid_search import (
    parse_alpha_grid,
    parse_confidence_grid,
    run_alpha_confidence_grid_search,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline 2D grid search on PMI alpha and confidence threshold."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Existing backtest CSV with raw_signal_logprob_* and realized_return columns.",
    )
    parser.add_argument(
        "--alphas",
        default="0,0.25,0.5,0.75,1.0,1.25",
        help='Comma-separated alpha list, e.g. "0,0.25,0.5,1.0".',
    )
    parser.add_argument(
        "--levels",
        default="0.30,0.35,0.40,0.45,0.50,0.55",
        help='Comma-separated confidence threshold list, e.g. "0.3,0.35,0.4".',
    )
    parser.add_argument(
        "--output",
        default="output/alpha_confidence_grid_search.csv",
        help="Where to save the summary CSV.",
    )
    parser.add_argument(
        "--detailed-output-dir",
        default=None,
        help="Optional directory for per-combination detailed backtest CSVs.",
    )
    args = parser.parse_args()

    summary = run_alpha_confidence_grid_search(
        args.input,
        alphas=parse_alpha_grid(args.alphas),
        confidence_levels=parse_confidence_grid(args.levels),
        output_path=args.output,
        detailed_output_dir=args.detailed_output_dir,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
