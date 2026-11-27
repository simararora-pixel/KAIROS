from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dashboard.core import load_workspace, forecast, outcome, explanation, evaluation, FEATURES, SCORES

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def w(): return load_workspace(ROOT)


def test_model_predictions_and_shap(w):
    m=w['bundle']['model']
    assert m.n_estimators==200 and m.n_features_in_==7
    for scope,frame in [('test',w['dataset'].query("Partition == 'test'")),('latest',w['inputs'])]:
        saved=w[scope]
        assert frame.Date.tolist()==saved.Date.tolist()
        np.testing.assert_array_equal(m.predict(frame[FEATURES]),saved.Prediction)
        np.testing.assert_allclose(m.predict_proba(frame[FEATURES]),saved[SCORES],atol=1e-12)
        shap=w[scope+'_shap'].groupby(['Date','Class']).agg(total=('SHAP','sum'),base=('Base_score','first'),score=('Model_score','first'),n=('Feature','count'))
        assert len(shap)==len(frame)*3 and shap.n.eq(7).all()
        np.testing.assert_allclose(shap.total+shap.base,shap.score,atol=1e-7)
        np.testing.assert_allclose(shap.score.to_numpy().reshape(-1,3),saved[SCORES],atol=1e-12)


def test_reveal_hide_and_date_selection(w):
    date='2024-07-29'; f=forecast(w,date)
    assert f['prediction']==0 and 'target' not in f and 'return_pct' not in f
    assert outcome(w,date,False) is None
    actual=outcome(w,date,True)
    assert actual['target']==-1 and not actual['correct'] and actual['return_pct'] < -3
    assert outcome(w,date,False) is None
    date2='2024-06-04'; assert forecast(w,date2)['prediction']==1
    assert outcome(w,date2,True)['target']==1
    ex=explanation(w,date,0); assert len(ex)==7
    with pytest.raises(ValueError): forecast(w,'1900-01-01')
    with pytest.raises(ValueError): outcome(w,w['latest'].Date.iloc[-1],True)


def test_metrics_and_data(w):
    assert [len(w[k]) for k in ['nifty','vix','market','dataset','test','validation','latest']]==[4965,4049,4050,4013,603,600,5]
    met,cm=evaluation(w,'test'); assert cm.tolist()==[[0,21,2],[40,480,18],[6,24,12]]
    assert met['Major Down recall']==0
    evaluation(w,'validation')
    assert w['dataset'][FEATURES].notna().all().all()
