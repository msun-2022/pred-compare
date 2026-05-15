"""Command-line interface for prediction-comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from prediction_comparison import __version__
from prediction_comparison.runner import run_comparison


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prediction-comparison",
        description=(
            "Compare prediction performance across multiple approaches. "
            "Generates metrics, pairwise tests, and 9 diagnostic plots."
        ),
    )
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    p.add_argument("--input", required=True,
                   help="Path to CSV with measured target, predictions, and features.")
    p.add_argument("--truth", required=True,
                   help="Column name of the measured target.")
    p.add_argument("--preds", nargs="+", required=True,
                   help="Column names for prediction approaches (>=2).")
    p.add_argument("--features", nargs="+", required=True,
                   help="Continuous feature columns (e.g. V1 V2 V3 V4).")
    p.add_argument("--outdir", default="results",
                   help="Output directory (default: results)")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap iterations for CIs (default: 1000)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--max-k", type=int, default=None,
                   help="Max k for top-k curve (default: N)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level and CI width (default: 0.05)")
    p.add_argument("--skip-radar", action="store_true",
                   help="Omit the radar plot (useful for <3 methods)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(input_path)
    try:
        run_comparison(
            df=df,
            truth=args.truth,
            preds=args.preds,
            features=args.features,
            outdir=args.outdir,
            n_boot=args.n_boot,
            seed=args.seed,
            max_k=args.max_k,
            alpha=args.alpha,
            skip_radar=args.skip_radar,
            verbose=not args.quiet,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
