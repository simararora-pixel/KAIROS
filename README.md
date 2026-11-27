# KAIROS

Classification of significant NIFTY 50 movements over five trading sessions using historical OHLC prices and India VIX.

## Research notebook

[`notebooks/KAIROS.ipynb`](notebooks/KAIROS.ipynb) contains the complete executable research workflow through target creation. Open it in Google Colab from the repository's `main` branch.

Place `KAIROS_NIFTY50_NSE_2006_2026.csv` and `KAIROS_INDIA_VIX_2008_2026.csv` in the working directory, or set `KAIROS_DATA_DIR` to the directory containing them. Run the notebook in order. It generates a six-column `KAIROS_COMBINED_CLEANED.csv` and a separate `KAIROS_VIX_IMPUTATION_AUDIT.csv`. The optional live acquisition code is not run by default.

## Project layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Research notebook |
| `src/preprocessing.py` | Reusable data preparation |
| `src/features.py` | Trailing market and VIX predictors |
| `src/targets.py` | Five-session classification labels |
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
- No model training or dashboard implementation yet.

## Limitations

The provisional ±3% threshold is a research choice. The historical snapshot has been examined in earlier experiments; the test period is a procedural holdout, not independent external validation. Data redistribution permissions remain unverified, so CSV files and notebook outputs are not committed. AI-assisted development and editing were used; the implementation and results require independent review.
