# V2 research results

Target: unchanged five-trading-session forward return, major below -3% or above +3%; otherwise ordinary. All later periods were inspected before V2. These are retrospective comparisons, not independent final evaluation.

## Directional experiment

Four chronological development folds with five-row gaps and explicit outcome-date assertions. The matched training set has 2,770 rows, after the additional 60-session warm-up. Validation and historical test remain 600 and 603 rows. Original seven vs expanded 22 predictors; direct and two-stage classification; logistic, forest, Extra Trees and histogram gradient boosting; weighted/unweighted; 36 decision rules. 32 fitted configurations, 1,152 configuration/rule combinations.

Selection prefers pooled development recall of at least 20% for both directions, then highest mean fold macro F1. Selected: original seven predictors, direct balanced Extra Trees (120 trees, depth 8, minimum leaf size 10, random seed 42), argmax. Mean CV macro F1 0.37506815163868346.

| Historical test metric | V1 | V2 |
|---|---:|---:|
| Accuracy | 0.8159 | 0.7877 |
| Macro F1 | 0.4091 | 0.4040 |
| Balanced accuracy | 0.3926 | 0.4025 |
| Down recall | 0 | 0.0870 |
| Up recall | 0.2857 | 0.2619 |
| Major correctly classified | 12 | 13 |
| Ordinary false alarms | 58 | 76 |

V1 remains the directional reference. No overall improvement established.

## Auxiliary magnitude task

Select by mean fold average precision, with warning threshold determined on training-fold scores. Selected unweighted logistic regression, StandardScaler, C=0.1, original seven predictors, threshold 0.20. Mean CV AP 0.2968507967318913.

Historical test: 65/603 event prevalence (0.1078), AP 0.2584, precision 0.4516, recall 0.2154, 14 correct warnings and 17 false alarms. VIX ranking alone AP 0.2966. Validation AP 0.2801 versus VIX 0.2523. This auxiliary model is experimental and does not establish superiority over simple VIX ranking. Its threshold and scores are not calibrated probabilities. Binary risk metrics cannot be compared directly to multiclass F1.

The original datasets were regenerated from the supplied raw snapshots, and locked V1 predictions reproduced. V2 uses causal-feature prefix checks and score/decision tests. Both exported V2 model bundles were reloaded and their outputs verified. Fresh independent data, calibration and prospective evaluation are required before any practical use.

Run notebook sections 18–19 to reproduce diagnostics and model artifacts locally. Data redistribution rights are unverified; raw market files, individual predictions, and model bundles are not committed.
