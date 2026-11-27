"""Freeze the reference model and experimental flow candidates without test tuning."""
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, recall_score

BASE = ['Return_1d','Return_5d','Volatility_20d','MA_20_gap_pct','Intraday_range_pct','VIX_Close','VIX_change_1d']
FII = ['FII_monthly_net','FII_three_month_net']
DII = ['DII_monthly_net','DII_three_month_net']
FEATURES = {'reference': BASE, 'fii': BASE+FII, 'all_four': BASE+FII+DII}
LABELS = [-1,0,1]


def freeze_models(datasets, monthly_z, risk_model, risk_spec):
    train = datasets['train'].copy()
    matched = train.join(monthly_z, on='Date', validate='one_to_one').dropna(subset=FII+DII)
    assert train.Date.is_unique and train.Date.is_monotonic_increasing
    assert train.Outcome_date.max() < datasets['validation'].Date.min()
    assert train.Outcome_date.max() < datasets['test'].Date.min()
    models = {}
    for name, cols in FEATURES.items():
        part = train if name == 'reference' else matched
        assert part[cols].notna().all().all()
        model = RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,criterion='gini',max_features='sqrt',bootstrap=True,class_weight='balanced',random_state=42,n_jobs=1)
        model.fit(part[cols],part.Target)
        assert model.classes_.tolist() == LABELS
        models[name] = model
    assert list(risk_spec['features']) == BASE and list(risk_model.classes_) == [0,1]
    return {'models':models,'features':FEATURES,'risk_model':risk_model,'risk_threshold':float(risk_spec['threshold']),
            'roles':{'reference':'Directional reference; original locked V1','fii':'Experimental CV-leading monthly-flow candidate','all_four':'Experimental four-dataset candidate'},
            'sklearn_version':sklearn.__version__,'training_rows':{'reference':len(train),'fii':len(matched),'all_four':len(matched)},
            'training_last_outcome':str(train.Outcome_date.max().date()),'flow_months':sorted(monthly_z.dropna(subset=FII+DII).Flow_month.unique().tolist()),
            'horizon':5,'threshold_pct':3.0}


def predict_frame(bundle, frame, name='reference'):
    if name not in bundle['models']:
        raise ValueError('Unknown model')
    cols = bundle['features'][name]
    if not set(['Date']+cols).issubset(frame.columns):
        raise ValueError('Missing required input columns')
    dates = pd.to_datetime(frame.Date, errors='raise')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be unique and chronological')
    if name != 'reference':
        if 'Flow_month' not in frame:
            raise ValueError('Flow_month is required for institutional models')
        expected = (dates.dt.to_period('M')-2).astype(str)
        if not frame.Flow_month.astype(str).eq(expected).all() or not expected.isin(bundle['flow_months']).all():
            raise ValueError('Unavailable or wrongly lagged flow month')
    if not np.isfinite(frame[cols].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite inputs')
    model=bundle['models'][name]
    prob=model.predict_proba(frame[cols])
    np.testing.assert_allclose(prob.sum(axis=1),1)
    risk=bundle['risk_model'].predict_proba(frame[BASE])[:,1]
    result=pd.DataFrame({'Date':dates,'Model':name,'Prediction':model.predict(frame[cols]),'Score_Down':prob[:,0],'Score_Ordinary':prob[:,1],'Score_Up':prob[:,2],'Risk_score':risk,'Risk_warning':risk>=bundle['risk_threshold']},index=frame.index)
    assert np.array_equal(result.Prediction.to_numpy(),np.array(LABELS)[prob.argmax(axis=1)])
    return result


def consolidate_metrics(bundle, datasets, monthly_z):
    rows=[]; predictions=[]
    for period in ('validation','test'):
        p=datasets[period].copy()
        matched=p.join(monthly_z,on='Date',validate='one_to_one').dropna(subset=FII+DII)
        for scope, part, names in [('full',p,['reference']),('matched',matched,list(FEATURES))]:
            for name in names:
                pred=predict_frame(bundle,part,name)
                y=part.Target
                r=recall_score(y,pred.Prediction,labels=LABELS,average=None,zero_division=0)
                rows.append({'Period':period,'Scope':scope,'Model':name,'Rows':len(y),'Accuracy':accuracy_score(y,pred.Prediction),'Macro F1':f1_score(y,pred.Prediction,labels=LABELS,average='macro',zero_division=0),'Balanced accuracy':balanced_accuracy_score(y,pred.Prediction),'Down recall':r[0],'Up recall':r[2]})
                pred['Target']=y; pred['Outcome_date']=part.Outcome_date; pred['Forward_return_5d_pct']=part.Forward_return_5d_pct; pred['Scope']=scope; pred['Period']=period; predictions.append(pred)
    return pd.DataFrame(rows),pd.concat(predictions,ignore_index=True)
