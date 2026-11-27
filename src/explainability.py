"""Post-hoc explanations for the locked Random Forest; never fit or tune it."""
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score


def global_importance(model, X, y, repeats=20):
    scorer = make_scorer(f1_score, labels=[-1, 0, 1], average='macro', zero_division=0)
    p = permutation_importance(model, X, y, scoring=scorer, n_repeats=repeats,
                               random_state=42, n_jobs=1)
    return pd.DataFrame({'Feature': X.columns, 'Impurity importance': model.feature_importances_,
                         'Permutation mean': p.importances_mean,
                         'Permutation SD': p.importances_std}).sort_values('Permutation mean', ascending=False)


def explain_rows(model, frame, features):
    X = frame[features]
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent', model_output='raw')
    values = np.asarray(explainer.shap_values(X))
    base = np.asarray(explainer.expected_value)
    assert model.classes_.tolist() == [-1, 0, 1]
    assert values.shape == (len(X), len(features), 3) and base.shape == (3,)
    probabilities = model.predict_proba(X)
    np.testing.assert_allclose(base + values.sum(axis=1), probabilities, atol=1e-7, rtol=0)
    rows = []
    for k, label in enumerate(model.classes_):
        for j, feature in enumerate(features):
            rows.append(pd.DataFrame({'Date': frame.Date.to_numpy(), 'Class': label,
                'Feature': feature, 'Value': X[feature].to_numpy(), 'SHAP': values[:, j, k],
                'Base_score': base[k], 'Model_score': probabilities[:, k]}))
    return pd.concat(rows, ignore_index=True)


def error_summary(records):
    result = []
    for label in [-1, 0, 1]:
        subset = records.loc[records.Target.eq(label)]
        result.append({'Actual class': label, 'Rows': len(subset),
                       'Predicted down': int(subset.Prediction.eq(-1).sum()),
                       'Predicted ordinary': int(subset.Prediction.eq(0).sum()),
                       'Predicted up': int(subset.Prediction.eq(1).sum()),
                       'Correct': int(subset.Target.eq(subset.Prediction).sum())})
    return pd.DataFrame(result)
