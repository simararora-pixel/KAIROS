"""Decision tree candidates for market movement classification."""
from sklearn.tree import DecisionTreeClassifier


def build_tree(class_weight=None, max_depth=4, min_samples_leaf=30):
    return DecisionTreeClassifier(criterion='gini', max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  class_weight=class_weight, random_state=42)
