"""Tests for the metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction_comparison.metrics import (
    bootstrap_metric,
    build_metrics_table,
    build_ranking_table,
    holm_bonferroni,
    ndcg_at_k,
    paired_wilcoxon_table,
    top_k_hit_rate,
)


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 100, 30)
    pred_cols = {
        "good": y_true + rng.normal(0, 1, 30),
        "ok": y_true + rng.normal(0, 5, 30),
        "bad": y_true + rng.normal(0, 15, 30),
    }
    return y_true, pred_cols


# ---------------------------------------------------------------------------
# top_k_hit_rate
# ---------------------------------------------------------------------------

def test_top_k_hit_rate_perfect():
    y = np.arange(10, dtype=float)
    assert top_k_hit_rate(y, y, 3) == 1.0


def test_top_k_hit_rate_random():
    y_true = np.array([5.0, 4, 3, 2, 1])
    y_pred = np.array([1.0, 2, 3, 4, 5])  # exact reverse
    assert top_k_hit_rate(y_true, y_pred, 1) == 0.0


def test_top_k_hit_rate_out_of_range():
    y = np.arange(5, dtype=float)
    assert np.isnan(top_k_hit_rate(y, y, 0))
    assert np.isnan(top_k_hit_rate(y, y, 100))


def test_top_k_hit_rate_full_k_is_one():
    y_true = np.array([1.0, 2, 3])
    y_pred = np.array([3.0, 2, 1])
    assert top_k_hit_rate(y_true, y_pred, 3) == 1.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

def test_ndcg_perfect_ranking():
    y = np.array([10.0, 8, 6, 4, 2])
    assert ndcg_at_k(y, y, 3) == pytest.approx(1.0)


def test_ndcg_handles_negative_targets():
    y_true = np.array([-5.0, -3, -1, 1, 3])
    result = ndcg_at_k(y_true, y_true, 3)
    assert 0 <= result <= 1


# ---------------------------------------------------------------------------
# bootstrap_metric
# ---------------------------------------------------------------------------

def test_bootstrap_metric_returns_ordered_ci(simple_data):
    y_true, pred_cols = simple_data
    lo, hi = bootstrap_metric(y_true, pred_cols["good"],
                              lambda yt, yp: float(np.sqrt(np.mean((yt - yp) ** 2))),
                              n_boot=200, seed=0)
    assert lo <= hi
    assert lo > 0


def test_bootstrap_metric_deterministic_with_seed(simple_data):
    y_true, pred_cols = simple_data
    fn = lambda yt, yp: float(np.mean((yt - yp) ** 2))
    a = bootstrap_metric(y_true, pred_cols["good"], fn, n_boot=100, seed=42)
    b = bootstrap_metric(y_true, pred_cols["good"], fn, n_boot=100, seed=42)
    assert a == b


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------

def test_holm_bonferroni_basic():
    pvals = np.array([0.01, 0.02, 0.04])
    adj = holm_bonferroni(pvals)
    # Smallest gets * m, then * (m-1), then * (m-2)
    assert adj[0] == pytest.approx(0.03)  # 0.01 * 3
    assert adj[1] == pytest.approx(0.04)  # 0.02 * 2
    assert adj[2] == pytest.approx(0.04)  # 0.04 * 1, but enforced monotone


def test_holm_bonferroni_caps_at_one():
    pvals = np.array([0.5, 0.6, 0.9])
    adj = holm_bonferroni(pvals)
    assert all(a <= 1.0 for a in adj)


def test_holm_bonferroni_monotone():
    """After sorting, adjusted p-values must be non-decreasing."""
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 0.1, 15)
    adj = holm_bonferroni(pvals)
    sorted_adj = adj[np.argsort(pvals)]
    assert np.all(np.diff(sorted_adj) >= -1e-12)


def test_holm_bonferroni_empty():
    adj = holm_bonferroni(np.array([]))
    assert len(adj) == 0


# ---------------------------------------------------------------------------
# build_metrics_table
# ---------------------------------------------------------------------------

def test_metrics_table_has_expected_columns(simple_data):
    y_true, pred_cols = simple_data
    df = build_metrics_table(y_true, pred_cols, n_boot=100, seed=0)
    expected = {
        "Approach", "RMSE", "MAE", "R2",
        "Pearson_r", "Spearman_rho", "Kendall_tau",
        "Bias_mean", "Bias_median",
    }
    assert expected.issubset(df.columns)
    for m in ["RMSE", "MAE", "R2", "Spearman_rho"]:
        assert f"{m}_CI_lo" in df.columns
        assert f"{m}_CI_hi" in df.columns


def test_metrics_table_row_per_method(simple_data):
    y_true, pred_cols = simple_data
    df = build_metrics_table(y_true, pred_cols, n_boot=50, seed=0)
    assert len(df) == len(pred_cols)
    assert set(df["Approach"]) == set(pred_cols.keys())


def test_metrics_table_good_beats_bad(simple_data):
    y_true, pred_cols = simple_data
    df = build_metrics_table(y_true, pred_cols, n_boot=100, seed=0)
    rmse_good = df.loc[df["Approach"] == "good", "RMSE"].iloc[0]
    rmse_bad = df.loc[df["Approach"] == "bad", "RMSE"].iloc[0]
    assert rmse_good < rmse_bad


# ---------------------------------------------------------------------------
# paired_wilcoxon_table
# ---------------------------------------------------------------------------

def test_paired_wilcoxon_pair_count(simple_data):
    y_true, pred_cols = simple_data
    pw = paired_wilcoxon_table(y_true, pred_cols)
    # 3 methods => C(3, 2) = 3 pairs
    assert len(pw) == 3
    assert "p_adj_holm" in pw.columns
    assert "significant" in pw.columns


def test_paired_wilcoxon_with_many_methods():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 100, 40)
    pred_cols = {
        f"m{i}": y_true + rng.normal(0, 2 + i, 40) for i in range(6)
    }
    pw = paired_wilcoxon_table(y_true, pred_cols)
    # 6 methods => C(6, 2) = 15
    assert len(pw) == 15
    # All adjusted p-values are in [0, 1]
    assert (pw["p_adj_holm"] >= 0).all()
    assert (pw["p_adj_holm"] <= 1).all()


def test_paired_wilcoxon_identical_predictions():
    """When two predictions are identical, the test handles d ~= 0 gracefully."""
    y_true = np.arange(20, dtype=float)
    pred_cols = {"a": y_true + 0.5, "b": y_true + 0.5}
    pw = paired_wilcoxon_table(y_true, pred_cols)
    assert len(pw) == 1
    assert pw["p_value"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# build_ranking_table
# ---------------------------------------------------------------------------

def test_ranking_table_sorted_by_mean_rank(simple_data):
    y_true, pred_cols = simple_data
    metrics_df = build_metrics_table(y_true, pred_cols, n_boot=50, seed=0)
    ranks = build_ranking_table(metrics_df)
    assert ranks["mean_rank"].is_monotonic_increasing
    # First row should be "good"
    assert ranks["Approach"].iloc[0] == "good"


def test_ranking_table_rank_columns_are_integers(simple_data):
    y_true, pred_cols = simple_data
    metrics_df = build_metrics_table(y_true, pred_cols, n_boot=50, seed=0)
    ranks = build_ranking_table(metrics_df)
    rank_cols = [c for c in ranks.columns if c.startswith("rank_")]
    for c in rank_cols:
        assert ranks[c].dtype.kind in "iu"
