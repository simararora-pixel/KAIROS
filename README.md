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
