"""
CLI entrypoint for offline signal-confidence grid search.
"""

from __future__ import annotations

import argparse

from backtest.pmi_grid_search import (
    parse_confidence_grid,
    run_signal_confidence_grid_search,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline confidence-threshold grid search on an existing backtest CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Existing backtest CSV with raw_signal_logprob_* and realized_return columns.",
    )
    parser.add_argument(
        "--levels",
        default="0.30,0.35,0.40,0.45,0.50,0.55",
        help='Comma-separated confidence threshold list, e.g. "0.3,0.35,0.4".',
    )
    parser.add_argument(
        "--pmi-alpha",
        type=float,
        default=1.0,
        help="PMI alpha to hold fixed while sweeping confidence thresholds.",
    )
    parser.add_argument(
        "--output",
        default="output/signal_confidence_grid_search.csv",
        help="Where to save the summary CSV.",
    )
    parser.add_argument(
        "--detailed-output-dir",
        default=None,
        help="Optional directory for per-threshold detailed backtest CSVs.",
    )
    args = parser.parse_args()

    summary = run_signal_confidence_grid_search(
        args.input,
        confidence_levels=parse_confidence_grid(args.levels),
        pmi_alpha=args.pmi_alpha,
        output_path=args.output,
        detailed_output_dir=args.detailed_output_dir,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
