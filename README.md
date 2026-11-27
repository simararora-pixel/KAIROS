# KAIROS

Classification of significant NIFTY 50 movements over five trading sessions using historical OHLC prices and India VIX.

## Research notebook

[`notebooks/KAIROS.ipynb`](notebooks/KAIROS.ipynb) contains the complete executable research workflow through final test evaluation. Open it in Google Colab from the repository's `main` branch.

Place `KAIROS_NIFTY50_NSE_2006_2026.csv` and `KAIROS_INDIA_VIX_2008_2026.csv` in the working directory, or set `KAIROS_DATA_DIR` to the directory containing them. Run the notebook in order. It generates a six-column `KAIROS_COMBINED_CLEANED.csv` and a separate `KAIROS_VIX_IMPUTATION_AUDIT.csv`. The optional live acquisition code is not run by default.

## Project layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Research notebook |
| `src/preprocessing.py` | Reusable data preparation |
| `src/features.py` | Trailing market and VIX predictors |
| `src/targets.py` | Five-session classification labels |
| `src/dataset.py` | Partition-safe dataset assembly |
| `src/baseline.py` | Majority-class and momentum benchmarks |
| `src/logistic.py` | Standardized multinomial logistic regression |
| `src/decision_tree.py` | Gini-based decision tree candidates |
| `src/random_forest.py` | Bootstrap random forest candidates |
| `src/gradient_boosting.py` | Sequential multiclass gradient boosting |
| `src/tuning.py` | Training-only expanding-window grid search |
| `src/ablation.py` | Matched feature-set comparisons and CV selection |
| `src/final_evaluation.py` | Test prediction records and score checks |
| `tests/test_final_evaluation.py` | Prediction alignment and probability checks |
| `tests/test_ablation.py` | Feature-set and selection-independence tests |
| `docs/selected_model.json` | Locked configuration before final test scoring |
| `tests/test_tuning.py` | Fold gap, outcome date and grid checks |
| `tests/test_gradient_boosting.py` | Stage structure and sample weight checks |
| `tests/test_random_forest.py` | Probability aggregation and reproducibility checks |
| `tests/test_decision_tree.py` | Tree complexity and probability checks |
| `tests/test_logistic.py` | Training-only scaling and probability checks |
| `tests/test_baseline.py` | Benchmark and metric checks |
| `tests/test_targets.py` | Threshold and partition boundary tests |
| `requirements.txt` | Python dependencies |
| `data/` | Local datasets (not committed) |
| `models/` | Planned model artifacts |
| `dashboard/` | Planned Streamlit application |

## Current checkpoint

- Data preprocessing: 4,050 aligned observations.
- Training-period exploratory analysis and seven candidate features.
- Chronological 70/15/15 split, with five-session target boundary exclusions.
- 2,810 eligible training rows after feature warm-up and target filtering.
- Final labelled research dataset: 4,013 rows, 16 columns; training 2,810, validation 600, test 603.
- Seven model inputs, five unlabelled recent prediction rows.
- Section 9 exports all datasets and a preparation audit to `kairos_datasets/` (override with `KAIROS_OUTPUT_DIR`).
- Baseline models evaluated on validation only. Majority-class macro F1: 0.3133; momentum macro F1: 0.3338.
- Logistic regression validation macro F1: 0.3133 (unweighted) and 0.4385 (balanced); balanced accuracy: 0.4929 for the balanced model.
- Decision tree validation macro F1: 0.3828 (unweighted), 0.3882 (balanced), 0.3616 (unrestricted diagnostic).
- The unrestricted tree reached training macro F1 of 1.0 with depth 26, illustrating overfitting.
- Random forest validation macro F1: 0.3133 (unweighted), 0.4274 (balanced); balanced forest accuracy 0.7350 and balanced accuracy 0.4956.
- Gradient boosting validation macro F1: 0.3136 (unweighted), 0.4179 (balanced); balanced accuracy 0.5155 for the balanced model.
- Balanced logistic regression remains the leading initial candidate by validation macro F1 (0.4385).
- Hyperparameter search: 36 configurations, four chronological folds, 144 CV fits, five-session gaps with outcome-date checks. Each family winner is refitted on all training observations.
- Tuned validation macro F1: Logistic 0.4365, Tree 0.3760, Forest 0.4422, Boosting 0.4076. Tuned forest also has the highest mean CV macro F1 (0.3888).
- Section 15 exports the grid-search diagnostics to `KAIROS_CV_SEARCH.csv` in the dataset export directory.
- Ablation study: same 36-configuration grid repeated with five price-only features, using matched rows and folds. Across both grids, 72 configurations and 288 CV fits.
- Random forest with VIX selected by highest mean training-fold macro F1 (0.3887568264467185); price-only forest 0.3730. Validation macro F1: 0.4422 with VIX vs 0.4180 without VIX. All four tuned families improved on validation with VIX, but CV evidence is mixed across families.
- Final configuration locked: balanced random forest, 200 trees, depth 8, minimum leaf size 10, sqrt feature sampling, bootstrap enabled, random seed 42, all seven features. Fitted on original training rows only.
- Final test evaluation (603 rows, forecast dates 2024-06-04 to 2026-11-06): selected forest macro F1 0.4091, accuracy 0.8159, balanced accuracy 0.3926; price-only forest macro F1 0.3839; momentum 0.3647; majority class 0.3143.
- Selected forest correctly classified 12 of 42 major rises and none of 23 major declines. It incorrectly flagged 58 ordinary observations as major. Configuration remains unchanged.
- Section 17 exports test prediction records and aggregate metrics locally. The repeated validation comparisons are exploratory; test outcomes overlap and scores are uncalibrated. Reliable trading performance is not established.
- Explainability, model packaging and dashboard implementation remain pending.

## Limitations

The provisional ±3% threshold is a research choice. The historical snapshot has been examined in earlier experiments; the test period is a procedural holdout, not independent external validation. Data redistribution permissions remain unverified, so CSV files and notebook outputs are not committed. AI-assisted development and editing were used; the implementation and results require independent review.
