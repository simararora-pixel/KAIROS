"""Validation benchmarks for five-session market movement classification."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)

CLASSES = [-1, 0, 1]
CLASS_NAMES = ['Major Down', 'Ordinary', 'Major Up']


def momentum_predict(X, threshold=3.0):
    """Classify trailing five-session return using the target thresholds."""
    if threshold <= 0:
        raise ValueError('Threshold must be positive')
    returns = X['Return_5d'].to_numpy(dtype=float)
    if not np.isfinite(returns).all():
        raise ValueError('Momentum inputs must be finite')
    boundary = np.isclose(np.abs(returns), threshold, rtol=0, atol=1e-10)
    return np.select([(returns < -threshold) & ~boundary,
                      (returns > threshold) & ~boundary], [-1, 1], default=0)


def evaluate_predictions(y_true, y_pred):
    """Score every target class, including classes never predicted."""
    report = classification_report(y_true, y_pred, labels=CLASSES,
                                   target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    summary = {'Accuracy': accuracy_score(y_true, y_pred),
               'Balanced accuracy': balanced_accuracy_score(y_true, y_pred),
               'Macro F1': f1_score(y_true, y_pred, labels=CLASSES,
                                    average='macro', zero_division=0),
               'Major Down recall': report['Major Down']['recall'],
               'Major Up recall': report['Major Up']['recall']}
    per_class = pd.DataFrame(report).T.loc[CLASS_NAMES,
                                         ['precision', 'recall', 'f1-score', 'support']]
    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES),
                          index=CLASS_NAMES, columns=CLASS_NAMES)
    return summary, per_class, matrix
