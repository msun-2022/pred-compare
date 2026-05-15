"""End-to-end example: generate synthetic data with 6 methods and run the pipeline.

Run from the repo root:
    python examples/run_example.py

Outputs land in ./example_results/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prediction_comparison import run_comparison

RNG = np.random.default_rng(42)


def _generate_truth(n: int) -> tuple[pd.DataFrame, np.ndarray]:
    """4 continuous features, nonlinear ground truth, plus mild noise."""
    V1 = RNG.uniform(10, 45, n)
    V2 = RNG.uniform(10, 40, n)
    V3 = RNG.uniform(8, 30, n)
    V4 = RNG.uniform(0, 2, n)
    features = pd.DataFrame({"V1": V1, "V2": V2, "V3": V3, "V4": V4})
    # Nonlinear, bounded target ~ "efficacy"
    y = (
        0.6 * V1
        + 0.4 * V2
        - 0.15 * (V3 - 18) ** 2
        + 12 * V4
        + 8 * np.sin(V1 / 6)
        + RNG.normal(0, 2, n)
    )
    y = np.clip(y, 0, 95)
    return features, y


def _make_predictions(y: np.ndarray, features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Synthesize 6 prediction methods with different failure modes."""
    n = len(y)
    return {
        # Near-perfect (small symmetric noise)
        "near_perfect": y + RNG.normal(0, 1.0, n),
        # Regression-to-mean shrinkage (the classic ensemble tilt)
        "shrunk": 0.85 * y + 0.15 * y.mean() + RNG.normal(0, 3.0, n),
        # GP-like: tight inside, two extreme outliers
        "gp_like": y + RNG.normal(0, 2.0, n),
        # Noisy baseline
        "noisy": y + RNG.normal(0, 10.0, n),
        # Biased high
        "biased_high": y + 5 + RNG.normal(0, 3.0, n),
        # Random rank but right magnitude
        "scrambled": y.mean() + RNG.normal(0, y.std(), n),
    }


def main() -> None:
    n = 50
    features, y = _generate_truth(n)
    preds = _make_predictions(y, features)

    # Inject GP-like edge failures: two points get huge errors
    edge_idx = np.argsort(features["V1"].values + features["V4"].values)[[0, -1]]
    preds["gp_like"][edge_idx[0]] -= 40
    preds["gp_like"][edge_idx[1]] += 45

    df = pd.concat([features, pd.Series(y, name="Y"), pd.DataFrame(preds)], axis=1)

    out = Path(__file__).parent.parent / "example_results"
    run_comparison(
        df=df,
        truth="Y",
        preds=list(preds.keys()),
        features=["V1", "V2", "V3", "V4"],
        outdir=out,
        n_boot=500,  # smaller for speed in the example
        seed=42,
    )


if __name__ == "__main__":
    main()
