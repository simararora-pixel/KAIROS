import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from src.baseline import momentum_predict, evaluate_predictions


def test_momentum_thresholds():
    X = pd.DataFrame({'Return_5d': [-4, -3, 0, 3, 4]})
    assert momentum_predict(X).tolist() == [-1, 0, 0, 0, 1]


def test_majority_and_metrics():
    X = pd.DataFrame({'Return_5d': [0., 1., 2., 3.]})
    y = [0, 0, -1, 1]
    model = DummyClassifier(strategy='most_frequent').fit(X, y)
    pred = model.predict(X)
    assert (pred == 0).all()
    summary, per_class, matrix = evaluate_predictions(y, pred)
    assert np.isclose(summary['Balanced accuracy'], 1/3)
    assert summary['Major Up recall'] == summary['Major Down recall'] == 0
    assert matrix.to_numpy().sum() == 4
