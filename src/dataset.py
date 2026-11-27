"""Assemble partition-safe research datasets from aligned prices and features."""
import pandas as pd
import numpy as np

FEATURE_COLUMNS = ['Return_1d', 'Return_5d', 'Volatility_20d', 'MA_20_gap_pct',
                   'Intraday_range_pct', 'VIX_Close', 'VIX_change_1d']
MARKET_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'VIX_Close']


def build_datasets(df, X, partitions, make_targets):
    assert df.index.is_unique and X.index.equals(df.index)
    assert df.Date.is_unique and df.Date.is_monotonic_increasing
    assert X.columns.tolist() == FEATURE_COLUMNS
    pd.testing.assert_series_equal(X.VIX_Close, df.VIX_Close)
    assert not np.isinf(X.to_numpy(dtype=float)).any()
    assert list(partitions) == ['train', 'validation', 'test']
    used = []
    datasets = {}
    audit = []
    for name, section in partitions.items():
        market = df.iloc[section]
        features = X.iloc[section]
        used.extend(market.index.tolist())
        labels = make_targets(market)
        assert labels.index.equals(market.index)
        feature_ok = features.notna().all(axis=1)
        target_ok = labels.Target.notna()
        eligible = feature_ok & target_ok
        joined = market[MARKET_COLUMNS].join(features.drop(columns='VIX_Close')).join(labels)
        joined['Partition'] = name
        part = joined.loc[eligible].copy()
        part['Target'] = part.Target.astype(int)
        assert not part.empty and part.notna().all().all()
        assert part.Target.isin([-1, 0, 1]).all()
        assert part.Outcome_date.gt(part.Date).all()
        assert part.Outcome_date.le(market.Date.max()).all()
        assert part.Date.is_unique and part.Date.is_monotonic_increasing
        datasets[name] = part
        audit.append({'Partition': name, 'Original': len(market),
                      'Unavailable_target': int((~target_ok).sum()),
                      'Unavailable_features_with_target': int((target_ok & ~feature_ok).sum()),
                      'Eligible': len(part)})
    assert used == df.index.tolist()
    final = pd.concat(datasets.values(), ignore_index=True)
    assert final.Date.is_unique and final.Date.is_monotonic_increasing
    assert len(final) == sum(len(part) for part in datasets.values())
    for left, right in [('train', 'validation'), ('validation', 'test')]:
        assert datasets[left].Outcome_date.max() < datasets[right].Date.min()
    recent = df.tail(5)
    recent_features = X.loc[recent.index]
    prediction = recent[['Date']].join(recent_features).dropna().copy()
    assert not {'Target', 'Outcome_date', 'Forward_return_5d_pct'}.intersection(prediction.columns)
    assert set(prediction.Date).isdisjoint(final.Date)
    return final, datasets, prediction, pd.DataFrame(audit)
