import pandas as pd
import numpy as np

def make_features(data):
    features = pd.DataFrame(index=data.index)
    close = data['Close']
    vix_close = data['VIX_Close']
    daily_return = close.pct_change(fill_method=None) * 100
    features['Return_1d'] = daily_return
    features['Return_5d'] = close.pct_change(5, fill_method=None) * 100
    features['Volatility_20d'] = daily_return.rolling(20).std()
    features['MA_20_gap_pct'] = (close / close.rolling(20).mean() - 1) * 100
    features['Intraday_range_pct'] = (data['High'] - data['Low']) / data['Open'] * 100
    features['VIX_Close'] = vix_close
    features['VIX_change_1d'] = vix_close.pct_change(fill_method=None) * 100
    features['VIX_missing'] = data['VIX_missing']
    # Mark changes involving an imputed VIX as missing rather than inventing a move.
    unreliable = data['VIX_missing'].eq(1) | data['VIX_missing'].shift(1, fill_value=0).eq(1)
    features.loc[unreliable, 'VIX_change_1d'] = np.nan
    return features
