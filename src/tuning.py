"""Chronological hyperparameter search with fold-local fitting and weighting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import f1_score

TUNING_GRIDS = {
    'Logistic': {'C': [0.01, 0.1, 1.0, 10.0], 'class_weight': [None, 'balanced']},
    'Tree': {'max_depth': [3, 5, 7], 'min_samples_leaf': [20, 50], 'class_weight': [None, 'balanced']},
    'Forest': {'max_depth': [4, 8], 'min_samples_leaf': [10, 30], 'class_weight': [None, 'balanced']},
    'Boosting': {'n_estimators': [50, 100], 'max_depth': [1, 2], 'balanced': [False, True]}
}


def chronological_folds(training, n_splits=4, test_size=400, gap=5):
    dates = pd.to_datetime(training['Date'])
    outcomes = pd.to_datetime(training['Outcome_date'])
    assert dates.is_unique and dates.is_monotonic_increasing
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    folds = list(splitter.split(training))
    for tr, va in folds:
        assert tr[-1] + gap < va[0]
        assert outcomes.iloc[tr].max() < dates.iloc[va].min()
        assert set(training.Target.iloc[tr].unique()) == {-1, 0, 1}
        assert set(training.Target.iloc[va].unique()) == {-1, 0, 1}
    return folds


def search_family(name, X, y, folds, fit_candidate):
    rows = []
    for params in ParameterGrid(TUNING_GRIDS[name]):
        scores = []
        for tr, va in folds:
            model = fit_candidate(name, params, X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[va])
            scores.append(f1_score(y.iloc[va], pred, labels=[-1, 0, 1],
                                   average='macro', zero_division=0))
        rows.append({'Family': name, 'Parameters': params,
                     **{f'Fold {i+1}': score for i, score in enumerate(scores)},
                     'Mean CV macro F1': np.mean(scores), 'SD CV macro F1': np.std(scores)})
    return pd.DataFrame(rows).sort_values('Mean CV macro F1', ascending=False, kind='stable').reset_index(drop=True)
