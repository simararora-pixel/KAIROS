import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid
from src.tuning import TUNING_GRIDS, chronological_folds


def test_chronological_folds_and_grid():
    dates = pd.date_range('2020-01-01', periods=105)
    data = pd.DataFrame({'Date': dates[:100], 'Outcome_date': dates[5:],
                         'Target': np.resize([-1, 0, 1], 100)})
    folds = chronological_folds(data, n_splits=3, test_size=20, gap=5)
    assert [len(tr) for tr, va in folds] == [35, 55, 75]
    for tr, va in folds:
        assert len(va) == 20
        assert data.Outcome_date.iloc[tr].max() < data.Date.iloc[va].min()
    with pytest.raises(AssertionError):
        chronological_folds(data, n_splits=3, test_size=20, gap=4)
    assert sum(len(ParameterGrid(grid)) for grid in TUNING_GRIDS.values()) == 36
