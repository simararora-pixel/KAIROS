"""Matched feature-set comparisons and deterministic CV selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def feature_sets(columns):
    full = list(columns)
    assert {'VIX_Close', 'VIX_change_1d'}.issubset(full)
    price = [c for c in full if c not in {'VIX_Close', 'VIX_change_1d'}]
    assert len(full) == 7 and len(price) == 5
    return {'With VIX': full, 'Without VIX': price}


def select_by_cv(results):
    assert results['Mean CV macro F1'].notna().all()
    return results.sort_values('Mean CV macro F1', ascending=False, kind='stable').iloc[0]
