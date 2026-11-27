# KAIROS

Classification of significant NIFTY 50 movements over five trading sessions using historical OHLC prices and India VIX.

## Research notebook

[`notebooks/KAIROS.ipynb`](notebooks/KAIROS.ipynb) contains the complete executable research workflow through multiclass logistic regression. Open it in Google Colab from the repository's `main` branch.

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
- Further model comparison, tuning, final test evaluation and dashboard implementation remain pending.

## Limitations

The provisional ±3% threshold is a research choice. The historical snapshot has been examined in earlier experiments; the test period is a procedural holdout, not independent external validation. Data redistribution permissions remain unverified, so CSV files and notebook outputs are not committed. AI-assisted development and editing were used; the implementation and results require independent review.
