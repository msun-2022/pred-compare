"""Tests for plot generation and end-to-end runner."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np
import pandas as pd
import pytest

from prediction_comparison.runner import run_comparison


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 40
    V1 = rng.uniform(10, 40, n)
    V2 = rng.uniform(10, 40, n)
    V3 = rng.uniform(10, 30, n)
    V4 = rng.uniform(0, 2, n)
    y = 0.5 * V1 + 0.3 * V2 + 10 * V4 + rng.normal(0, 2, n)
    df = pd.DataFrame({
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "Y": y,
        "m1": y + rng.normal(0, 1, n),
        "m2": y + rng.normal(0, 3, n),
        "m3": y + rng.normal(0, 8, n),
    })
    return df


def test_run_comparison_end_to_end(synthetic_df, tmp_path):
    result = run_comparison(
        df=synthetic_df,
        truth="Y",
        preds=["m1", "m2", "m3"],
        features=["V1", "V2", "V3", "V4"],
        outdir=tmp_path,
        n_boot=50,
        seed=0,
        verbose=False,
    )
    # Tables returned
    assert "metrics" in result
    assert "wilcoxon" in result
    assert "ranking" in result

    # CSVs written
    assert (tmp_path / "metrics_table.csv").exists()
    assert (tmp_path / "paired_wilcoxon.csv").exists()
    assert (tmp_path / "ranking_summary.csv").exists()

    # PNGs written
    expected = [
        "01_pred_vs_measured.png",
        "02_residuals_vs_measured.png",
        "03_metrics_with_CIs.png",
        "04_topk_hitrate.png",
        "05_error_distributions.png",
        "06_residuals_vs_components.png",
        "07_pca_error_map.png",
        "08_pairwise_rmse_heatmap.png",
        "09_radar_metrics.png",
    ]
    for f in expected:
        assert (tmp_path / f).exists(), f"missing {f}"


def test_run_comparison_with_many_methods(tmp_path):
    """Smoke test: pipeline works with 8 methods."""
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame({
        "V1": rng.uniform(0, 50, n),
        "V2": rng.uniform(0, 50, n),
        "V3": rng.uniform(0, 50, n),
        "V4": rng.uniform(0, 2, n),
    })
    df["Y"] = 0.7 * df["V1"] + 0.3 * df["V2"] + rng.normal(0, 3, n)
    method_names = []
    for i in range(8):
        name = f"method_{i}"
        df[name] = df["Y"] + rng.normal(0, 1 + i * 0.5, n)
        method_names.append(name)

    result = run_comparison(
        df=df,
        truth="Y",
        preds=method_names,
        features=["V1", "V2", "V3", "V4"],
        outdir=tmp_path,
        n_boot=30,
        seed=0,
        verbose=False,
    )
    assert len(result["metrics"]) == 8
    # 8 methods => C(8, 2) = 28 pairs
    assert len(result["wilcoxon"]) == 28


def test_run_comparison_rejects_single_method(synthetic_df, tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        run_comparison(
            df=synthetic_df,
            truth="Y",
            preds=["m1"],
            features=["V1", "V2"],
            outdir=tmp_path,
            n_boot=10,
            verbose=False,
        )


def test_run_comparison_rejects_missing_columns(synthetic_df, tmp_path):
    with pytest.raises(ValueError, match="Missing columns"):
        run_comparison(
            df=synthetic_df,
            truth="Y",
            preds=["m1", "nonexistent"],
            features=["V1"],
            outdir=tmp_path,
            n_boot=10,
            verbose=False,
        )


def test_run_comparison_handles_nans(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "V1": rng.uniform(0, 30, n),
        "V2": rng.uniform(0, 30, n),
        "Y": rng.uniform(0, 100, n),
    })
    df["m1"] = df["Y"] + rng.normal(0, 1, n)
    df["m2"] = df["Y"] + rng.normal(0, 3, n)
    # Inject NaNs
    df.loc[0, "m1"] = np.nan
    df.loc[5, "V1"] = np.nan

    result = run_comparison(
        df=df, truth="Y", preds=["m1", "m2"],
        features=["V1", "V2"], outdir=tmp_path,
        n_boot=30, verbose=False,
    )
    # Should have run successfully on the cleaned subset
    assert len(result["metrics"]) == 2
