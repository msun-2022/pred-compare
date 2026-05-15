"""Visualization functions for prediction comparison. All plots scale to many methods."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from prediction_comparison.metrics import ndcg_at_k, top_k_hit_rate


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _grid_dims(n: int) -> tuple[int, int]:
    if n <= 2:
        return 1, n
    if n <= 4:
        return 2, 2
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _panel_size(n: int, base: float = 4.0, floor: float = 3.0) -> float:
    """Per-panel figure size shrinks gracefully as method count grows."""
    return max(floor, base - 0.2 * max(0, n - 4))


def _palette(n: int) -> list[tuple]:
    """Return n distinct (color, linestyle) pairs. Works up to ~40 methods."""
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if n > 10:
        extra = plt.cm.Set2(np.linspace(0, 1, 8))
        base_colors = np.vstack([base_colors, extra])
    linestyles = ['-', '--', '-.', ':']
    return [
        (base_colors[i % len(base_colors)],
         linestyles[(i // len(base_colors)) % len(linestyles)])
        for i in range(n)
    ]


def _save(fig, outpath, bbox_inches=None):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches=bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 01 - Predicted vs measured
# ---------------------------------------------------------------------------

def plot_pred_vs_measured(y_true, pred_cols: Mapping[str, np.ndarray], outpath):
    """Scatter of predicted vs measured, one panel per method, with y=x line."""
    n = len(pred_cols)
    nr, nc = _grid_dims(n)
    ps = _panel_size(n)
    fig, axes = plt.subplots(nr, nc, figsize=(ps * nc, ps * nr), squeeze=False)
    lo = min(y_true.min(), min(p.min() for p in pred_cols.values()))
    hi = max(y_true.max(), max(p.max() for p in pred_cols.values()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    for ax, (name, y_pred) in zip(axes.flat, pred_cols.items()):
        ax.scatter(y_true, y_pred, alpha=0.7, s=24,
                   edgecolor="white", linewidth=0.5)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "k--", lw=1, alpha=0.6)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{name}\nRMSE={rmse:.3g}, R²={r2:.3f}", fontsize=10)
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, outpath)


# ---------------------------------------------------------------------------
# Plot 02 - Residuals vs measured
# ---------------------------------------------------------------------------

def plot_residuals_vs_measured(y_true, pred_cols, outpath):
    n = len(pred_cols)
    nr, nc = _grid_dims(n)
    ps = _panel_size(n)
    fig, axes = plt.subplots(nr, nc,
                             figsize=(ps * nc, ps * 0.9 * nr), squeeze=False)
    for ax, (name, y_pred) in zip(axes.flat, pred_cols.items()):
        resid = y_pred - y_true
        ax.scatter(y_true, resid, alpha=0.7, s=24,
                   edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="k", lw=1, linestyle="--", alpha=0.6)
        ax.set_title(
            f"{name}  (bias={resid.mean():.3g}, med={np.median(resid):.3g})",
            fontsize=10,
        )
        ax.set_xlabel("Measured")
        ax.set_ylabel("Residual (pred − meas)")
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, outpath)


# ---------------------------------------------------------------------------
# Plot 03 - Metrics with CIs
# ---------------------------------------------------------------------------

def plot_metrics_with_cis(metrics_df: pd.DataFrame, outpath,
                          metrics_to_plot=("RMSE", "R2", "Spearman_rho")):
    n_methods = len(metrics_df)
    fig_w = max(5 * len(metrics_to_plot), 1.2 * n_methods * len(metrics_to_plot))
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(fig_w, 4.5))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    x = np.arange(n_methods)
    rot = 45 if n_methods >= 5 else 30
    for ax, m in zip(axes, metrics_to_plot):
        vals = metrics_df[m].to_numpy()
        lo = metrics_df[f"{m}_CI_lo"].to_numpy()
        hi = metrics_df[f"{m}_CI_hi"].to_numpy()
        err = np.vstack([vals - lo, hi - vals])
        ax.bar(x, vals, yerr=err, capsize=4, color="#4C72B0",
               edgecolor="black", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["Approach"], rotation=rot, ha="right")
        ax.set_title(f"{m} (95% bootstrap CI)")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, outpath)


# ---------------------------------------------------------------------------
# Plot 04 - Top-k hit rate and NDCG
# ---------------------------------------------------------------------------

def plot_topk_hitrate(y_true, pred_cols, outpath, max_k: int | None = None):
    n = len(y_true)
    if max_k is None:
        max_k = n
    ks = np.arange(1, max_k + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    pairs = _palette(len(pred_cols))
    for (name, y_pred), (color, ls) in zip(pred_cols.items(), pairs):
        hits = [top_k_hit_rate(y_true, y_pred, k) for k in ks]
        ndcg = [ndcg_at_k(y_true, y_pred, k) for k in ks]
        ax1.plot(ks, hits, marker="o", markersize=2.8, linestyle=ls,
                 color=color, label=name, lw=1.5)
        ax2.plot(ks, ndcg, marker="o", markersize=2.8, linestyle=ls,
                 color=color, label=name, lw=1.5)
    for ax, title, ylabel in [(ax1, "Top-k hit rate", "Top-k hit rate"),
                              (ax2, "NDCG@k", "NDCG@k")]:
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    if len(pred_cols) > 6:
        ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=8, frameon=False)
    else:
        ax1.legend(fontsize=9)
        ax2.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, outpath, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Plot 05 - Error distributions
# ---------------------------------------------------------------------------

def plot_error_distributions(y_true, pred_cols, outpath):
    names = list(pred_cols)
    abs_err = [np.abs(pred_cols[n] - y_true) for n in names]
    fig_w = max(1.2 * len(names) + 2, 6)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    parts = ax.violinplot(abs_err, showmeans=False,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.5)
    ax.boxplot(
        abs_err, widths=0.18, patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black"),
        medianprops=dict(color="firebrick", lw=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    rot = 45 if len(names) >= 5 else 30
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=rot, ha="right")
    ax.set_ylabel("|Residual|")
    ax.set_title("Absolute-error distribution per approach")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, outpath)


# ---------------------------------------------------------------------------
# Plot 06 - Residuals vs features
# ---------------------------------------------------------------------------

def plot_residuals_vs_components(y_true, pred_cols, features_df: pd.DataFrame, outpath):
    feat_names = list(features_df.columns)
    n_approaches = len(pred_cols)
    fig_h = min(2.6 * n_approaches, 20)
    fig_w = min(3.0 * len(feat_names), 16)
    fig, axes = plt.subplots(n_approaches, len(feat_names),
                             figsize=(fig_w, fig_h), squeeze=False)
    for i, (name, y_pred) in enumerate(pred_cols.items()):
        resid = y_pred - y_true
        for j, feat in enumerate(feat_names):
            ax = axes[i, j]
            ax.scatter(features_df[feat], resid, alpha=0.7, s=18,
                       edgecolor="white", linewidth=0.4)
            ax.axhline(0, color="k", lw=0.8, linestyle="--", alpha=0.6)
            if i == 0:
                ax.set_title(feat)
            if j == 0:
                ax.set_ylabel(f"{name}\nresidual", fontsize=9)
            if i == n_approaches - 1:
                ax.set_xlabel(feat)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outpath)


# ---------------------------------------------------------------------------
# Plot 07 - PCA error map
# ---------------------------------------------------------------------------

def plot_pca_error_map(y_true, pred_cols, features_df: pd.DataFrame, outpath):
    X = StandardScaler().fit_transform(features_df.values)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    n = len(pred_cols)
    nr, nc = _grid_dims(n)
    ps = _panel_size(n, base=4.5, floor=3.2)
    fig, axes = plt.subplots(nr, nc, figsize=(ps * nc, ps * nr), squeeze=False)
    abs_errs = {name: np.abs(p - y_true) for name, p in pred_cols.items()}
    vmax = max(e.max() for e in abs_errs.values()) or 1.0
    for ax, (name, ae) in zip(axes.flat, abs_errs.items()):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=ae, cmap="viridis",
                        vmin=0, vmax=vmax, s=45,
                        edgecolor="black", linewidth=0.4)
        ax.set_title(f"{name}  |error| on feature PCA", fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
        fig.colorbar(sc, ax=ax, label="|residual|")
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, outpath)


# ---------------------------------------------------------------------------
# Plot 08 - Pairwise RMSE heatmap
# ---------------------------------------------------------------------------

def plot_pairwise_rmse_heatmap(y_true, pred_cols, wilcoxon_df: pd.DataFrame,
                               outpath, alpha: float = 0.05):
    """Heatmap of RMSE(row) - RMSE(col), with Holm-significance asterisks."""
    names = list(pred_cols)
    n = len(names)
    rmse = {nm: float(np.sqrt(mean_squared_error(y_true, p)))
            for nm, p in pred_cols.items()}
    diff = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            diff[i, j] = rmse[a] - rmse[b]

    sig_lookup = {}
    if len(wilcoxon_df) > 0:
        for _, r in wilcoxon_df.iterrows():
            sig_lookup[(r["A"], r["B"])] = r["p_adj_holm"]
            sig_lookup[(r["B"], r["A"])] = r["p_adj_holm"]

    fig_w = max(6, 0.7 * n + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    vmax = np.abs(diff).max() or 1.0
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title(
        "Pairwise RMSE difference: RMSE(row) − RMSE(col)\n"
        f"(blue = row better; * = Holm-adjusted p < {alpha})"
    )
    for i in range(n):
        for j in range(n):
            txt = f"{diff[i, j]:+.2f}"
            p = sig_lookup.get((names[i], names[j]))
            if p is not None and p < alpha and i != j:
                txt += "*"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if abs(diff[i, j]) > 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, label="ΔRMSE")
    fig.tight_layout()
    _save(fig, outpath)


# ---------------------------------------------------------------------------
# Plot 09 - Radar of normalized metrics
# ---------------------------------------------------------------------------

def plot_radar_metrics(metrics_df: pd.DataFrame, outpath):
    """Polar plot. Each metric is normalized so 1 = best across the methods shown.

    Note: normalization is *relative*. A method at 1.0 is best in this comparison,
    not best in absolute terms.
    """
    metric_specs = [
        ("RMSE", "lower"),
        ("MAE", "lower"),
        ("R2", "higher"),
        ("Spearman_rho", "higher"),
        ("Kendall_tau", "higher"),
        ("Bias_abs", "lower"),
    ]
    df = metrics_df.copy()
    df["Bias_abs"] = df["Bias_mean"].abs()

    norm = {}
    for name, direction in metric_specs:
        vals = df[name].to_numpy(dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if vmax == vmin:
            norm[name] = np.ones_like(vals)
        elif direction == "lower":
            norm[name] = (vmax - vals) / (vmax - vmin)
        else:
            norm[name] = (vals - vmin) / (vmax - vmin)

    labels = [m for m, _ in metric_specs]
    n_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    pairs = _palette(len(df))
    for i, ((color, ls), (_, row)) in enumerate(zip(pairs, df.iterrows())):
        vals = [norm[m][i] for m, _ in metric_specs]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linestyle=ls, lw=1.8,
                label=row["Approach"])
        ax.fill(angles, vals, color=color, alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("Normalized metrics (1 = best across methods)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05),
              fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, outpath, bbox_inches="tight")
