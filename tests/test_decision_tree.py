import numpy as np
import pandas as pd
from src.decision_tree import build_tree


def test_tree_limits_and_probability_order():
    X = pd.DataFrame({'a': np.arange(120), 'b': np.tile([0, 1], 60)})
    y = np.repeat([-1, 0, 1], 40)
    for weight in (None, 'balanced'):
        model = build_tree(class_weight=weight).fit(X, y)
        assert model.get_depth() <= 4
        leaves = model.tree_.children_left == -1
        assert model.tree_.n_node_samples[leaves].min() >= 30
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        np.testing.assert_array_equal(model.predict(X), model.classes_[proba.argmax(axis=1)])
