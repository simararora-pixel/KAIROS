import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from src.gradient_boosting import build_boosting, fit_boosting


def test_weighted_multiclass_boosting():
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(150, 7)))
    y = np.repeat([-1, 0, 1], [20, 100, 30])
    weights = compute_sample_weight('balanced', y)
    np.testing.assert_allclose([weights[y == k].sum() for k in [-1, 0, 1]], [50, 50, 50])
    for balanced in (False, True):
        model = fit_boosting(build_boosting(n_estimators=5, min_samples_leaf=5), X, y, balanced)
        assert model.estimators_.shape == (5, 3)
        assert all(t.get_depth() <= 2 for t in model.estimators_.ravel())
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        np.testing.assert_array_equal(model.predict(X), model.classes_[proba.argmax(axis=1)])
        np.testing.assert_array_equal(list(model.staged_predict(X))[-1], model.predict(X))
