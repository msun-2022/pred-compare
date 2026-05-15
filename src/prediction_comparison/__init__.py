"""Prediction performance comparison toolkit for multi-method evaluation."""

from prediction_comparison.metrics import (
    bootstrap_metric,
    build_metrics_table,
    build_ranking_table,
    holm_bonferroni,
    ndcg_at_k,
    paired_wilcoxon_table,
    top_k_hit_rate,
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
from prediction_comparison.runner import run_comparison

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "run_comparison",
    # Metrics
    "bootstrap_metric",
    "build_metrics_table",
    "build_ranking_table",
    "holm_bonferroni",
    "ndcg_at_k",
    "paired_wilcoxon_table",
    "top_k_hit_rate",
    # Plots
    "plot_pred_vs_measured",
    "plot_residuals_vs_measured",
    "plot_metrics_with_cis",
    "plot_topk_hitrate",
    "plot_error_distributions",
    "plot_residuals_vs_components",
    "plot_pca_error_map",
    "plot_pairwise_rmse_heatmap",
    "plot_radar_metrics",
]
