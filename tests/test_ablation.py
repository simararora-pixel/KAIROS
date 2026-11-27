import pandas as pd
from src.ablation import feature_sets, select_by_cv


def test_feature_sets_and_selection_ignores_validation():
    full = ['Return_1d', 'Return_5d', 'Volatility_20d', 'MA_20_gap_pct', 'Intraday_range_pct', 'VIX_Close', 'VIX_change_1d']
    sets = feature_sets(full)
    assert sets['With VIX'] == full
    assert sets['Without VIX'] == full[:5]
    results = pd.DataFrame({'Family': ['A', 'B'], 'Mean CV macro F1': [0.4, 0.3], 'Macro F1': [0.1, 0.9]})
    assert select_by_cv(results)['Family'] == 'A'
    results['Macro F1'] = [0.9, 0.1]
    assert select_by_cv(results)['Family'] == 'A'
