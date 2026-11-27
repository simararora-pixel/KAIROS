"""Test-period prediction records for a previously selected classifier."""
import numpy as np
import pandas as pd


def prediction_records(model, test, features):
    X_test = test[features]
    assert model.classes_.tolist() == [-1, 0, 1]
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    assert np.isfinite(proba).all() and (proba >= 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    np.testing.assert_array_equal(pred, model.classes_[proba.argmax(axis=1)])
    records = test[['Date', 'Outcome_date', 'Forward_return_5d_pct', 'Target']].copy()
    records['Prediction'] = pred
    records[['Score_Down', 'Score_Ordinary', 'Score_Up']] = proba
    records['Correct'] = records.Target.eq(records.Prediction)
    return records
