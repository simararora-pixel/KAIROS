"""Multiclass logistic regression candidates for KAIROS."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic(class_weight=None):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(solver='lbfgs', C=1.0,
                                          max_iter=2000, class_weight=class_weight))
    ])
