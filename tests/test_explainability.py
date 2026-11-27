import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.explainability import explain_rows, global_importance, error_summary


def test_explanation_contract():
    x = pd.DataFrame(np.random.default_rng(42).normal(size=(90, 2)), columns=['a', 'b'])
    x['Date'] = pd.date_range('2020-01-01', periods=len(x))
    y = pd.Series(np.tile([-1, 0, 1], 30))
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(x[['a', 'b']], y)
    before = model.predict_proba(x[['a', 'b']])
    s = explain_rows(model, x, ['a', 'b'])
    assert len(s) == 90 * 2 * 3
    z = s.groupby(['Date', 'Class']).agg(total=('SHAP', 'sum'), base=('Base_score', 'first'), score=('Model_score', 'first'))
    np.testing.assert_allclose(z.total + z.base, z.score, atol=1e-7)
    assert len(global_importance(model, x[['a', 'b']], y, repeats=2)) == 2
    np.testing.assert_array_equal(before, model.predict_proba(x[['a', 'b']]))
    summary = error_summary(pd.DataFrame({'Target': y, 'Prediction': model.predict(x[['a', 'b']])}))
    assert summary.Rows.sum() == len(y)
