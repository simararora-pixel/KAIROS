"""Gradient boosting candidates for KAIROS."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


def build_boosting(n_estimators=100, learning_rate=0.05, max_depth=2, min_samples_leaf=30):
    return GradientBoostingClassifier(loss='log_loss', n_estimators=n_estimators,
                                      learning_rate=learning_rate, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf, subsample=1.0,
                                      random_state=42)


def fit_boosting(model, X, y, balanced=False):
    weights = compute_sample_weight('balanced', y) if balanced else None
    return model.fit(X, y, sample_weight=weights)
