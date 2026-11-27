# KAIROS

UML501 machine learning project: experimental classification of five-session NIFTY 50 movements using NIFTY 50 prices and India VIX.

## Run in Colab

Open [notebooks/KAIROS.ipynb](notebooks/KAIROS.ipynb) from the main branch. Upload only these two original source files to the notebook working directory (or set KAIROS_DATA_DIR):

- KAIROS_NIFTY50_NSE_2006_2026.csv
- KAIROS_INDIA_VIX_2008_2026.csv

Run the notebook in order. Optional live acquisition is not executed by default. For exact model reproduction, use scikit-learn 1.8.0, pandas 2.2.3 and NumPy 2.3.5. See requirements.txt.

## Single model

One balanced RandomForestClassifier (200 trees, depth 8, minimum leaf 10, Gini, sqrt feature sampling, bootstrap, seed 42). Seven trailing features: daily and five-session returns, 20-session volatility, 20-session moving-average gap, intraday range, VIX close and VIX change. No FII/DII or separate risk model. Majority-class and momentum rules are evaluation benchmarks only, not deployed estimators.

Target: five-trading-session net return below -3% (Major Down), above +3% (Major Up), otherwise Ordinary. Data split chronologically; outcome labels do not cross partition boundaries. 4,050 aligned market observations; 4,013 labelled rows: 2,810 training, 600 validation, 603 historical test.

## Reproduced performance

| Metric | Validation | Historical test |
|---|---:|---:|
| Accuracy | 0.7717 | 0.8159 |
| Macro F1 | 0.4422 | 0.4091 |
| Balanced accuracy | 0.4910 | 0.3926 |
| Major Down recall | 0.3714 | 0 |
| Major Up recall | 0.2727 | 0.2857 |

All 603 historical test predictions and their class scores reproduced the locked V1 reference. The majority rule has 0.8922 test accuracy and 0.3143 macro F1. The historical period is already inspected; this is not independent validation. The model missed all 23 major declines and is not reliable for trading. Class scores are uncalibrated. The target windows overlap.

## Outputs

The notebook exports datasets, metrics, prediction records, and KAIROS_REFERENCE.joblib under kairos_datasets/ (override with KAIROS_OUTPUT_DIR). It saves and reloads exactly one trained estimator and verifies equal predictions. For feature-ready CSV inference:

```bash
python predict.py --input example_inputs.csv --output predictions.csv
```

The input must contain Date and the seven named features. Use the same scikit-learn version as the bundle. Raw market data and model binaries remain local. Earlier experiments remain recoverable via GitHub history; they are not part of the active model. AI-assisted development was used; independently review results.

## Explainable AI

Section 14 adds impurity importance, validation permutation importance (20 repeats, macro F1), and path-dependent Tree SHAP for the unchanged single forest. VIX Close (0.0418) and trailing 20-session volatility (0.0238) rank highest by mean validation permutation importance. Test and latest snapshot SHAP values reconstruct all class scores; the notebook checks additivity and reproduces the same 603 test predictions. Exports are stored under `kairos_datasets/xai/`. These explanations describe the model, not market causality; permutation importance is affected by temporal dependence and correlated features. The missed-decline case on 2024-07-29 received an ordinary score of 0.8553 but realized a -3.1426% forward return. 

## Local Streamlit dashboard

Run the Colab notebook and download its exported `KAIROS_FINAL_DATASET.csv`. From the repository directory on your computer:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload the combined CSV in the sidebar (or put it in kairos_datasets/). Eight tabs cover the market snapshot, dataset, technical indicators, historical prediction, reveal/hide replay, performance, SHAP, and methodology. The app reconstructs the same fixed forest from training rows only and caches it; no tuning or additional estimator is used. No live feed. Actual outcomes in replay are hidden by default; choose a date and tick/untick Reveal actual outcome. Other analytics tabs show historical results openly. Browser and real-data integration testing are still required.

## Local dashboard

Run `python -m streamlit run dashboard/app.py` from the repository root after installing requirements. The eight native tabs cover overview, datasets, market analysis, prediction engine, historical simulator, performance, explainable AI and methodology. Exactly one saved Random Forest is used; the simulator uses only the eligible historical test inputs and reveals actual outcomes via a checkbox. Latest forecasts are unlabelled historical snapshot observations, not live signals.

Use the complete downloadable local dashboard package, or put the two original raw CSVs under `data/` and run the notebook with KAIROS_DATA_DIR pointing there and KAIROS_OUTPUT_DIR pointing to the repository-root `kairos_datasets/`. All required CSVs are now exported directly into `kairos_datasets/`. The repository does not include raw market data or fitted model binaries. Use trusted local artifacts and the pinned scikit-learn version.

The three dashboard backend tests passed using the supplied snapshot, including identical saved predictions and SHAP scores, metrics reconciliation and reveal/hide logic. All eight tabs' Python rendering paths were exercised against a mock Streamlit interface. Full Streamlit server and browser verification is pending: installation was unavailable in the development environment. All 19 weekend-dated observations have supporting special-session evidence; full official price/calendar reconciliation remains pending. No model settings or data were changed for this dashboard. The application is educational and offline after dependency installation.

## Double audit

Run `python -m dashboard.audit` and `python -m pytest -q tests/test_dashboard.py` with the local data and notebook exports. Independent calculations reproduced all 28,091 feature values, 4,013 labels and outcome dates, 1,203 validation/test predictions and scores, and evaluation metrics. All 33 notebook code cells and three backend tests passed after fixes. The final labelled dataset, saved forest and metrics/prediction CSVs are byte-identical to the previous package. The exploratory threshold boundary now matches target creation; the two dashboard CSVs are exported directly; the SHAP class selector uses date/scope-specific identity; obsolete launch instructions were removed and root app.py forwards to dashboard/app.py. Browser testing and complete external archive reconciliation remain pending. See docs/AUDIT.md.
