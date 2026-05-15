"""Orchestrator: run the full comparison and write all outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from prediction_comparison.metrics import (
    build_metrics_table,
    build_ranking_table,
    paired_wilcoxon_table,
)
from prediction_comparison.plots import (
    plot_error_distributions,
    plot_metrics_with_cis,
    plot_pairwise_rmse_heatmap,
    plot_pca_error_map,
    plot_pred_vs_measured,
    plot_radar_metrics,
    plot_residuals_vs_components,
    plot_residuals_vs_measured,
    plot_topk_hitrate,
)


def _validate(df: pd.DataFrame, truth: str, preds: Sequence[str],
              features: Sequence[str]) -> pd.DataFrame:
    """Drop NaN rows in required columns; validate column presence and method count."""
    missing = [c for c in [truth, *preds, *features] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(preds) < 2:
        raise ValueError("Need at least 2 prediction methods to compare.")
    needed = [truth, *preds, *features]
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    if len(df) < before:
        print(f"[info] Dropped {before - len(df)} rows with NaNs.")
    if len(df) < 5:
        raise ValueError("Need at least 5 rows after NaN filtering.")
    return df


def run_comparison(
    df: pd.DataFrame,
    truth: str,
    preds: Sequence[str],
    features: Sequence[str],
    outdir: str | Path,
    n_boot: int = 1000,
    seed: int = 0,
    max_k: int | None = None,
    alpha: float = 0.05,
    skip_radar: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the full comparison pipeline.

    Parameters
    ----------
    df : DataFrame
        One row per item (e.g., formulation). Must contain truth, preds, features.
    truth : str
        Column name of the measured target.
    preds : sequence of str
        Column names for the prediction approaches.
    features : sequence of str
        Continuous feature columns used for the residual-vs-feature and PCA plots.
    outdir : path
        Where to write CSVs and PNGs.
    n_boot : int
        Bootstrap iterations for CIs.
    seed : int
        RNG seed for the bootstrap.
    max_k : int, optional
        Max k for the top-k curve (defaults to N).
    alpha : float
        Significance level for pairwise tests and CI width.
    skip_radar : bool
        If True, omit the radar plot.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys 'metrics', 'wilcoxon', 'ranking' (pandas DataFrames) and
    'outdir' (Path).
    """
    df = _validate(df, truth, preds, list(features))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    y_true = df[truth].to_numpy(dtype=float)
    pred_cols = {name: df[name].to_numpy(dtype=float) for name in preds}
    features_df = df[list(features)].copy()
    n_methods = len(pred_cols)

    log = print if verbose else (lambda *_a, **_k: None)
    log(f"Comparing {n_methods} methods on {len(y_true)} items.\n")

    # Tables
    log("[1/10] Metrics + bootstrap CIs...")
    metrics_df = build_metrics_table(y_true, pred_cols,
                                     n_boot=n_boot, seed=seed, alpha=alpha)
    metrics_df.to_csv(outdir / "metrics_table.csv", index=False)

    log("[2/10] Pairwise Wilcoxon + Holm correction...")
    pw = paired_wilcoxon_table(y_true, pred_cols, alpha=alpha)
    pw.to_csv(outdir / "paired_wilcoxon.csv", index=False)

    log("[3/10] Ranking summary...")
    rank_df = build_ranking_table(metrics_df)
    rank_df.to_csv(outdir / "ranking_summary.csv", index=False)

    if verbose:
        display_cols = ["Approach", "RMSE", "MAE", "R2",
                        "Spearman_rho", "Bias_mean", "Bias_median"]
        print("\n=== Metrics ===")
        print(metrics_df[display_cols].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n=== Pairwise (after Holm correction) ===")
        print(pw[["A", "B", "p_value", "p_adj_holm", "significant"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print("\n=== Ranking summary ===")
        print(rank_df.to_string(index=False))

    # Plots
    log("\n[4/10] Pred vs measured...")
    plot_pred_vs_measured(y_true, pred_cols, outdir / "01_pred_vs_measured.png")

    log("[5/10] Residuals vs measured...")
    plot_residuals_vs_measured(y_true, pred_cols,
                               outdir / "02_residuals_vs_measured.png")

    log("[6/10] Metrics with CIs...")
    plot_metrics_with_cis(metrics_df, outdir / "03_metrics_with_CIs.png")

    log("[7/10] Top-k hit rate + NDCG...")
    plot_topk_hitrate(y_true, pred_cols, outdir / "04_topk_hitrate.png",
                      max_k=max_k)

    log("[8/10] Error distributions...")
    plot_error_distributions(y_true, pred_cols,
                             outdir / "05_error_distributions.png")

    log("[9/10] Residuals vs features + PCA error map...")
    plot_residuals_vs_components(y_true, pred_cols, features_df,
                                 outdir / "06_residuals_vs_components.png")
    plot_pca_error_map(y_true, pred_cols, features_df,
                       outdir / "07_pca_error_map.png")

    log("[10/10] Pairwise heatmap + radar...")
    plot_pairwise_rmse_heatmap(y_true, pred_cols, pw,
                               outdir / "08_pairwise_rmse_heatmap.png",
                               alpha=alpha)
    if not skip_radar and n_methods >= 3:
        plot_radar_metrics(metrics_df, outdir / "09_radar_metrics.png")

    log(f"\nDone. Outputs in: {outdir.resolve()}")
    return {
        "metrics": metrics_df,
        "wilcoxon": pw,
        "ranking": rank_df,
        "outdir": outdir,
    }
