import numpy as np
import pandas as pd
from src.random_forest import build_forest


def test_forest_probability_average_and_reproducibility():
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(150, 7)))
    y = np.tile([-1, 0, 1], 50)
    forest = build_forest(class_weight='balanced', n_estimators=12, min_samples_leaf=3).fit(X, y)
    other = build_forest(class_weight='balanced', n_estimators=12, min_samples_leaf=3).fit(X, y)
    proba = forest.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    np.testing.assert_allclose(proba, other.predict_proba(X))
    np.testing.assert_allclose(proba, np.mean([tree.predict_proba(X.to_numpy()) for tree in forest.estimators_], axis=0))
    assert all(tree.get_depth() <= 6 and tree.max_features_ == 2 for tree in forest.estimators_)
