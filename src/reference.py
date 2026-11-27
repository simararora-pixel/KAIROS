"""The single KAIROS directional model and inference contract."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = ['Return_1d','Return_5d','Volatility_20d','MA_20_gap_pct','Intraday_range_pct','VIX_Close','VIX_change_1d']


def build_reference():
    return RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                  criterion='gini', max_features='sqrt', bootstrap=True,
                                  class_weight='balanced', random_state=42, n_jobs=1)


def predict_reference(model, frame):
    if not set(['Date'] + FEATURES).issubset(frame.columns):
        raise ValueError('Missing required columns')
    dates = pd.to_datetime(frame.Date, errors='raise')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be unique and chronological')
    inputs = frame[FEATURES].astype(float)
    if not np.isfinite(inputs.to_numpy()).all():
        raise ValueError('Nonfinite inputs')
    assert model.classes_.tolist() == [-1, 0, 1]
    p = model.predict_proba(inputs)
    pred = model.predict(inputs)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_array_equal(pred, model.classes_[p.argmax(axis=1)])
    return pd.DataFrame({'Date':dates, 'Prediction':pred, 'Score_Down':p[:,0],
                         'Score_Ordinary':p[:,1], 'Score_Up':p[:,2]}, index=frame.index)
