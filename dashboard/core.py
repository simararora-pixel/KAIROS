"""Data and inference for the single-model KAIROS dashboard."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, recall_score

FEATURES = ['Return_1d','Return_5d','Volatility_20d','MA_20_gap_pct','Intraday_range_pct','VIX_Close','VIX_change_1d']
LABELS = {-1:'Major Down',0:'Ordinary',1:'Major Up'}
SCORES = ['Score_Down','Score_Ordinary','Score_Up']
NAMES = dict(zip(FEATURES,['1-session return','5-session return','20-session volatility','20-session MA gap','Intraday range','India VIX close','VIX daily change']))


def load_workspace(root):
    root = Path(root)
    d = root / 'kairos_datasets'
    files = {'market':'KAIROS_COMBINED_CLEANED.csv','dataset':'KAIROS_FINAL_DATASET.csv',
             'test':'KAIROS_TEST_PREDICTIONS.csv','latest':'KAIROS_LATEST_PREDICTIONS.csv',
             'inputs':'KAIROS_PREDICTION_INPUTS.csv','validation':'KAIROS_VALIDATION_PREDICTIONS.csv',
             'metrics':'KAIROS_METRICS.csv','audit':'KAIROS_DATASET_AUDIT.csv',
             'imputation':'KAIROS_VIX_IMPUTATION_AUDIT.csv','importance':'xai/KAIROS_FEATURE_IMPORTANCE.csv',
             'test_shap':'xai/KAIROS_TEST_SHAP.csv','latest_shap':'xai/KAIROS_LATEST_SHAP.csv'}
    w = {k:pd.read_csv(d/f) for k,f in files.items()}
    for k in ('nifty','vix'):
        f = {'nifty':'KAIROS_NIFTY50_NSE_2006_2026.csv','vix':'KAIROS_INDIA_VIX_2008_2026.csv'}[k]
        w[k] = pd.read_csv(root/'data'/f)
    w['manifest'] = json.loads((d/'KAIROS_MODEL_MANIFEST.json').read_text())
    if w['manifest']['sklearn_version'] != sklearn.__version__:
        raise RuntimeError('Install scikit-learn '+w['manifest']['sklearn_version']+' before loading the trusted local model.')
    w['bundle'] = joblib.load(d/'KAIROS_REFERENCE.joblib')
    assert w['bundle']['features'] == FEATURES and w['manifest']['features'] == FEATURES
    assert w['bundle']['model'].classes_.tolist() == [-1,0,1]
    for key in ('market','dataset','test','latest','inputs','validation','nifty','vix'):
        assert w[key].Date.is_unique and w[key].Date.is_monotonic_increasing
    return w


def forecast(w, date, latest=False):
    """Return only known-at-forecast inputs and scores, never actual outcomes."""
    frame = w['inputs'] if latest else w['dataset'].loc[w['dataset'].Partition.eq('test')]
    row = frame.loc[frame.Date.eq(date),FEATURES]
    if len(row)!=1 or not np.isfinite(row.to_numpy()).all():
        raise ValueError('No eligible forecast for this date')
    model = w['bundle']['model']
    p = model.predict_proba(row)[0]
    assert np.isclose(p.sum(),1)
    return {'date':date,'prediction':int(model.predict(row)[0]),'scores':p,'features':row.iloc[0]}


def outcome(w,date,reveal=False):
    """Return actual outcome only when the reveal control is active."""
    if not reveal:
        return None
    row = w['test'].loc[w['test'].Date.eq(date)]
    if len(row)!=1:
        raise ValueError('No scored test observation for this date')
    r = row.iloc[0]
    return {'date':r.Outcome_date,'target':int(r.Target),'return_pct':float(r.Forward_return_5d_pct),'correct':bool(r.Correct)}


def explanation(w,date,label,latest=False):
    t = w['latest_shap'] if latest else w['test_shap']
    a = t.loc[t.Date.eq(date)&t.Class.eq(label)].copy()
    assert len(a)==len(FEATURES) and set(a.Feature)==set(FEATURES)
    assert np.isclose(a.Base_score.iloc[0]+a.SHAP.sum(),a.Model_score.iloc[0],atol=1e-7)
    a['Feature name'] = a.Feature.map(NAMES)
    return a.sort_values('SHAP')


def evaluation(w,period):
    p = w[period]
    y, pred = p.Target, p.Prediction
    r = recall_score(y,pred,labels=[-1,0,1],average=None,zero_division=0)
    actual = {'Accuracy':accuracy_score(y,pred),'Balanced accuracy':balanced_accuracy_score(y,pred),
              'Macro F1':f1_score(y,pred,labels=[-1,0,1],average='macro',zero_division=0),
              'Major Down recall':r[0],'Major Up recall':r[2]}
    saved = w['metrics'].loc[w['metrics'].Period.eq(period)&w['metrics'].Model.eq('Random Forest')].iloc[0]
    np.testing.assert_allclose(list(actual.values()),saved[list(actual)].astype(float),atol=1e-12)
    return actual,confusion_matrix(y,pred,labels=[-1,0,1])
