"""Five-session classification labels, with partition boundary protection."""
import numpy as np
import pandas as pd


def make_targets(data, horizon=5, threshold=3.0):
    if horizon < 1 or threshold <= 0:
        raise ValueError('horizon and threshold must be positive')
    if not data.Date.is_monotonic_increasing or not data.Date.is_unique:
        raise ValueError('Dates must be unique and sorted')
    forward = (data.Close.shift(-horizon) / data.Close - 1) * 100
    boundary = np.isclose(forward.abs(), threshold, rtol=0, atol=1e-10)
    target = pd.Series(pd.NA, index=data.index, dtype='Int64')
    target.loc[forward.notna()] = 0
    target.loc[(forward > threshold) & ~boundary] = 1
    target.loc[(forward < -threshold) & ~boundary] = -1
    return pd.DataFrame({'Outcome_date': data.Date.shift(-horizon),
                         'Forward_return_5d_pct': forward, 'Target': target})
