import numpy as np
import pandas as pd
from src.logistic import build_logistic


def test_training_only_scaling_and_probabilities():
    X = pd.DataFrame({'a': [-3., -2., 0., 1., 3., 4.], 'b': [0., 1., 2., 3., 4., 5.]})
    y = [-1, -1, 0, 0, 1, 1]
    future = pd.DataFrame({'a': [100., 200.], 'b': [10., 30.]})
    for weight in (None, 'balanced'):
        model = build_logistic(weight).fit(X, y)
        np.testing.assert_allclose(model.named_steps['scaler'].mean_, X.mean())
        proba = model.predict_proba(future)
        assert proba.shape == (2, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        np.testing.assert_array_equal(model.predict(future), model.classes_[proba.argmax(axis=1)])
        np.testing.assert_allclose(model.named_steps['scaler'].mean_, X.mean())
