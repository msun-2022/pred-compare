# Prediction Comparison

Compare prediction performance across **multiple methods** on the same items, with proper statistics and diagnostic plots. Designed for evaluating regression-style predictors (efficacy, yield, score, etc.) on small to medium datasets where you have measured truth values and predictions from several approaches.

## Why?

When you have one set of items (formulations, compounds, configurations) with measured outcomes and predicted values from several methods, you typically want to know:

- Which method is most accurate? (RMSE, MAE)
- Which method ranks items correctly? (Spearman, top-k hit rate)
- Are the differences statistically meaningful? (paired Wilcoxon with Holm correction)
- Where in feature space does each method fail?
- Is the difference between two methods worth caring about, or within noise?

This toolkit answers all of those in one command and produces nine diagnostic plots.

## Installation

```bash
git clone https://github.com/msun-2022/pred-compare.git
cd pred-compare
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick start

### Input data format

A CSV with one row per item containing:

| Column | Description |
|---|---|
| (truth) | Measured target value (column name passed via `--truth`) |
| (preds) | One column per prediction method (names passed via `--preds`) |
| (features) | Continuous input features (names passed via `--features`) |

Example:

```csv
id,V1,V2,V3,V4,Y,method_a,method_b,method_c
1,12.3,20.1,15.7,0.5,67.2,65.8,69.1,71.0
2,18.0,25.4,11.2,1.1,82.5,80.4,79.9,76.3
...
```

### Command-line

```bash
prediction_comparison \
  --input data.csv \
  --truth Y \
  --preds method_a method_b method_c method_d method_e \
  --features V1 V2 V3 V4 \
  --outdir results/
```

### Python API

```python
import pandas as pd
from prediction_comparison import run_comparison

df = pd.read_csv("data.csv")
result = run_comparison(
    df=df,
    truth="Y",
    preds=["method_a", "method_b", "method_c", "method_d", "method_e"],
    features=["V1", "V2", "V3", "V4"],
    outdir="results/",
)

# Programmatic access
print(result["metrics"])   # DataFrame with metrics + bootstrap CIs
print(result["wilcoxon"])  # Pairwise tests with Holm-adjusted p-values
print(result["ranking"])   # Mean rank across metrics
```

## Outputs

In your `outdir`:

### Tables (CSV)

- **`metrics_table.csv`** — RMSE, MAE, R², Pearson, Spearman, Kendall, bias (mean and median), each with 95% bootstrap CI
- **`paired_wilcoxon.csv`** — All pairwise Wilcoxon signed-rank tests on squared errors, with raw and Holm-adjusted p-values
- **`ranking_summary.csv`** — Each method's rank on each metric, plus mean rank and best-metric count

### Plots (PNG)

| File | What it shows |
|---|---|
| `01_pred_vs_measured.png` | Scatter of predicted vs measured per method, with y=x reference |
| `02_residuals_vs_measured.png` | Residual plots — reveals tilt, heteroscedasticity, outliers |
| `03_metrics_with_CIs.png` | Bar charts of RMSE / R² / Spearman with bootstrap CIs |
| `04_topk_hitrate.png` | Top-k hit rate and NDCG@k curves (for prioritization use cases) |
| `05_error_distributions.png` | Violin + box plot of `|residual|` per method |
| `06_residuals_vs_components.png` | Residuals vs each feature, per method (detects feature-dependent bias) |
| `07_pca_error_map.png` | Feature-space PCA colored by per-method `|residual|` |
| `08_pairwise_rmse_heatmap.png` | RMSE-difference matrix with Holm-significance markers |
| `09_radar_metrics.png` | Polar plot of normalized metrics (requires ≥3 methods) |

## Scaling to many methods

The toolkit handles 2 to ~40 methods gracefully:

- Figure layouts auto-scale (panel size adjusts with method count)
- Top-k legend moves outside the plot at 7+ methods
- Color palette + line styles give up to 40 distinguishable line plots
- Holm-Bonferroni correction protects against pairwise inflation (with 10 methods, that's 45 pairwise tests — uncorrected, you'd see false positives by chance)

## Reading the outputs

Once the plots are generated, a typical reading order:

1. **`03_metrics_with_CIs.png`** — Get the headline ranking, note where CIs overlap (those differences aren't real)
2. **`02_residuals_vs_measured.png`** — Check whether top methods are well-calibrated or just got lucky on averages
3. **`04_topk_hitrate.png`** — If your use case is prioritization, this matters more than RMSE
4. **`08_pairwise_rmse_heatmap.png`** — See which pairwise differences survive multiple-testing correction
5. **`07_pca_error_map.png`** — Find where in feature space each method fails

## Caveats and best practices

**Evaluation protocol matters more than method choice.** The toolkit measures whatever you give it. If your predictions were generated in-sample (model saw the same points it's predicting on), the metrics will be optimistic. Always use held-out predictions or proper cross-validation.

**Small N gives wide CIs.** With N < 20, expect bootstrap CIs to overlap heavily and pairwise tests to lose power. The toolkit will report this honestly — interpret accordingly.

**The radar plot is relative.** A method at 1.0 on an axis is best *among the methods in this comparison*, not best in absolute terms. Always cross-reference with `metrics_table.csv`.

**Multiple-testing matters once you have 5+ methods.** With 5 methods you have 10 pairwise tests; with 10 methods, 45. Without Holm correction, you'd see "significant" results by chance. The toolkit applies Holm-Bonferroni automatically.

## Example

See [`examples/`](examples/) for a runnable example with synthetic data.

```bash
python examples/run_example.py
```

This generates a synthetic dataset with 4 features and 6 prediction methods of varying quality, then runs the full comparison.

## Contributing

Issues and pull requests welcome. Please run the tests before submitting a PR:

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
```


