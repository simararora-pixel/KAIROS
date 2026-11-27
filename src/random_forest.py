"""Random forest candidates for market movement classification."""
from sklearn.ensemble import RandomForestClassifier


def build_forest(class_weight=None, n_estimators=200, max_depth=6, min_samples_leaf=20):
    return RandomForestClassifier(n_estimators=n_estimators, criterion='gini',
                                  max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  max_features='sqrt', bootstrap=True,
                                  class_weight=class_weight, random_state=42, n_jobs=1)
