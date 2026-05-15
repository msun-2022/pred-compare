"""Numeric metrics and statistical tests for prediction comparison."""

from __future__ import annotations

import itertools
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def top_k_hit_rate(y_true: ArrayLike, y_pred: ArrayLike, k: int) -> float:
    """Fraction of true top-k captured by predicted top-k.

    Parameters
    ----------
    y_true : array
        Measured target values.
    y_pred : array
        Predicted values.
    k : int
        Cut-off. Must satisfy 0 < k <= len(y_true).

    Returns
    -------
    float in [0, 1], or NaN if k is out of range.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if k <= 0 or k > len(y_true):
        return np.nan
    true_top = set(np.argsort(-y_true)[:k])
    pred_top = set(np.argsort(-y_pred)[:k])
    return len(true_top & pred_top) / k


def ndcg_at_k(y_true: ArrayLike, y_pred: ArrayLike, k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k.

    Uses (y_true - min(y_true)) as gain so the metric is well-defined when
    targets can be negative. Returns NaN if all gains are zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if k <= 0 or k > len(y_true):
        return np.nan
    gains = y_true - y_true.min()
    order_pred = np.argsort(-y_pred)[:k]
    order_true = np.argsort(-y_true)[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order_pred] * discounts))
    idcg = float(np.sum(gains[order_true] * discounts))
    return dcg / idcg if idcg > 0 else np.nan


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_metric(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric_fn: Callable[[ArrayLike, ArrayLike], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap percentile CI for metric_fn(y_true, y_pred).

    Returns (lo, hi) at the (alpha/2, 1 - alpha/2) percentiles. Resampling
    is paired: the same row indices are used for both arrays.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            vals[i] = np.nan
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)
    return float(np.percentile(vals, lo_pct)), float(np.percentile(vals, hi_pct))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _metric_fns() -> dict[str, Callable]:
    return {
        "RMSE": lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": mean_absolute_error,
        "R2": r2_score,
        "Pearson_r": lambda yt, yp: float(stats.pearsonr(yt, yp).statistic),
        "Spearman_rho": lambda yt, yp: float(stats.spearmanr(yt, yp).statistic),
        "Kendall_tau": lambda yt, yp: float(stats.kendalltau(yt, yp).statistic),
        "Bias_mean": lambda yt, yp: float(np.mean(yp - yt)),
        "Bias_median": lambda yt, yp: float(np.median(yp - yt)),
    }


def build_metrics_table(
    y_true: ArrayLike,
    pred_cols: Mapping[str, ArrayLike],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per method; point estimates plus bootstrap CIs for each metric."""
    rows = []
    for name, y_pred in pred_cols.items():
        row = {"Approach": name}
        for m_name, fn in _metric_fns().items():
            row[m_name] = fn(y_true, y_pred)
            lo, hi = bootstrap_metric(
                y_true, y_pred, fn, n_boot=n_boot, seed=seed, alpha=alpha
            )
            row[f"{m_name}_CI_lo"] = lo
            row[f"{m_name}_CI_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multiple-testing correction and pairwise tests
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals: ArrayLike) -> np.ndarray:
    """Step-down Holm-Bonferroni adjusted p-values, enforcing monotonicity."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def paired_wilcoxon_table(
    y_true: ArrayLike,
    pred_cols: Mapping[str, ArrayLike],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """All pairwise Wilcoxon signed-rank tests on squared errors.

    Adds Holm-Bonferroni adjusted p-values across all pairs.
    """
    y_true = np.asarray(y_true)
    names = list(pred_cols)
    sq_err = {n: (np.asarray(pred_cols[n]) - y_true) ** 2 for n in names}
    rows = []
    for a, b in itertools.combinations(names, 2):
        d = sq_err[a] - sq_err[b]
        if np.allclose(d, 0):
            stat, p = np.nan, 1.0
        else:
            res = stats.wilcoxon(sq_err[a], sq_err[b], zero_method="wilcox")
            stat, p = float(res.statistic), float(res.pvalue)
        rows.append({
            "A": a,
            "B": b,
            "median_sq_err_A": float(np.median(sq_err[a])),
            "median_sq_err_B": float(np.median(sq_err[b])),
            "Wilcoxon_stat": stat,
            "p_value": p,
        })
    df = pd.DataFrame(rows)
    if len(df) >= 2:
        df["p_adj_holm"] = holm_bonferroni(df["p_value"].values)
    elif len(df) == 1:
        df["p_adj_holm"] = df["p_value"]
    else:
        df["p_adj_holm"] = pd.Series(dtype=float)
    df["significant"] = df["p_adj_holm"] < alpha
    return df


# ---------------------------------------------------------------------------
# Ranking summary
# ---------------------------------------------------------------------------

def build_ranking_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Rank each method on each metric (1 = best). Adds mean_rank and best_count."""
    lower_better = {"RMSE", "MAE", "Bias_mean_abs", "Bias_median_abs"}

    df = metrics_df.copy()
    df["Bias_mean_abs"] = df["Bias_mean"].abs()
    df["Bias_median_abs"] = df["Bias_median"].abs()

    out = pd.DataFrame({"Approach": df["Approach"]})
    metrics = [
        "RMSE", "MAE", "R2", "Pearson_r", "Spearman_rho",
        "Kendall_tau", "Bias_mean_abs", "Bias_median_abs",
    ]
    for m in metrics:
        ascending = m in lower_better
        out[f"rank_{m}"] = df[m].rank(method="min", ascending=ascending).astype(int)

    rank_cols = [c for c in out.columns if c.startswith("rank_")]
    out["mean_rank"] = out[rank_cols].mean(axis=1).round(3)
    out["best_count"] = (out[rank_cols] == 1).sum(axis=1)
    out = out.sort_values("mean_rank").reset_index(drop=True)
    return out
