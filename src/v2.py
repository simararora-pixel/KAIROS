"""KAIROS V2: causal features and chronological model development."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score, confusion_matrix, average_precision_score
from sklearn.model_selection import TimeSeriesSplit

LABELS = [-1, 0, 1]
V1 = ['Return_1d','Return_5d','Volatility_20d','MA_20_gap_pct','Intraday_range_pct','VIX_Close','VIX_change_1d']


def expanded_features(data, base_features):
    out = base_features.copy()
    c, h, l, o, v = [data[k] for k in ['Close','High','Low','Open','VIX_Close']]
    r = c.pct_change(fill_method=None) * 100
    for n in (3, 10, 20, 60):
        out[f'Return_{n}d'] = c.pct_change(n, fill_method=None) * 100
    for n in (5, 10, 60):
        out[f'Volatility_{n}d'] = r.rolling(n).std()
    out['Vol_ratio_5_20'] = out.Volatility_5d / out.Volatility_20d.replace(0, np.nan)
    out['MA_60_gap_pct'] = (c / c.rolling(60).mean() - 1) * 100
    out['Drawdown_20d'] = (c / c.rolling(20).max() - 1) * 100
    out['Overnight_gap'] = (o / c.shift(1) - 1) * 100
    out['Close_location'] = ((c-l)/(h-l).replace(0, np.nan)).fillna(0.5)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    out['ATR_14_pct'] = tr.rolling(14).mean() / c * 100
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out['RSI_14'] = 100*gain/(gain+loss).replace(0,np.nan)
    out['MACD_pct'] = (c.ewm(span=12, adjust=False).mean()-c.ewm(span=26, adjust=False).mean())/c*100
    # No additional VIX imputation: expanded predictors use only observed price history.
    return out.replace([np.inf,-np.inf],np.nan)


def estimator(kind, weighted):
    weight = 'balanced' if weighted else None
    if kind == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight=weight,max_iter=2000))
    if kind == 'forest':
        return RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,max_features='sqrt',class_weight=weight,random_state=42,n_jobs=1)
    if kind == 'extra':
        return ExtraTreesClassifier(n_estimators=120,max_depth=8,min_samples_leaf=10,class_weight=weight,random_state=42,n_jobs=1)
    if kind == 'hist':
        return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,min_samples_leaf=30,learning_rate=.05,l2_regularization=2,class_weight=weight,early_stopping=False,random_state=42)
    raise ValueError(kind)


def fit_scores(X,y,architecture,kind,weighted):
    if architecture == 'direct':
        model = estimator(kind,weighted).fit(X,y)
        assert list(model.classes_) == LABELS
        return (model,)
    event = estimator(kind,weighted).fit(X,y.ne(0).astype(int))
    direction = estimator(kind,weighted).fit(X.loc[y.ne(0)],y.loc[y.ne(0)].eq(1).astype(int))
    assert list(event.classes_) == list(direction.classes_) == [0,1]
    return event,direction


def scores(models,X):
    if len(models)==1:
        p=models[0].predict_proba(X)
    else:
        m,d=[model.predict_proba(X)[:,1] for model in models]
        p=np.column_stack([m*(1-d),1-m,m*d])
    assert np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.sum(axis=1),1)
    return p


def decide(p,rule):
    if rule[0]=='argmax':
        return np.array(LABELS)[p.argmax(axis=1)]
    _,event_threshold,up_threshold=rule
    event=p[:,0]+p[:,2]
    up=np.divide(p[:,2],event,out=np.full(len(p),.5),where=event>0)
    return np.where(event>=event_threshold,np.where(up>=up_threshold,1,-1),0)


def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p)
    recalls=recall_score(y,p,labels=LABELS,average=None,zero_division=0)
    precisions=precision_score(y,p,labels=LABELS,average=None,zero_division=0)
    return {'Macro F1':f1_score(y,p,labels=LABELS,average='macro',zero_division=0),'Accuracy':accuracy_score(y,p),'Balanced accuracy':balanced_accuracy_score(y,p),'Down recall':recalls[0],'Up recall':recalls[2],'Down precision':precisions[0],'Up precision':precisions[2],'Major correctly classified':int(((y!=0)&(y==p)).sum()),'Ordinary false alarms':int(((y==0)&(p!=0)).sum())}


def search_v2(X,y,training,feature_groups):
    folds=list(TimeSeriesSplit(n_splits=4,test_size=400,gap=5).split(X))
    for tr,va in folds:
        assert training.Outcome_date.iloc[tr].max()<training.Date.iloc[va].min()
        assert set(y.iloc[tr])==set(y.iloc[va])==set(LABELS)
    rules=[('argmax',)] + [('gate',m,d) for m in (.15,.25,.35,.45,.55,.65,.75) for d in (.35,.45,.5,.55,.65)]
    rows=[]
    for group,cols in feature_groups.items():
        for architecture in ('direct','two_stage'):
            for kind in ('logistic','forest','extra','hist'):
                for weighted in (False,True):
                    blocks=[]
                    for tr,va in folds:
                        fitted=fit_scores(X[cols].iloc[tr],y.iloc[tr],architecture,kind,weighted)
                        blocks.append((y.iloc[va],scores(fitted,X[cols].iloc[va])))
                    actual=pd.concat([b[0] for b in blocks])
                    for rule in rules:
                        preds=[decide(p,rule) for _,p in blocks]
                        fold_f1=[f1_score(a,p,labels=LABELS,average='macro',zero_division=0) for (a,_),p in zip(blocks,preds)]
                        pooled=metrics(actual,np.concatenate(preds))
                        rows.append({'features':group,'architecture':architecture,'algorithm':kind,'weighted':weighted,'rule':rule,'Mean CV macro F1':np.mean(fold_f1),'SD CV macro F1':np.std(fold_f1),**{f'Fold {i+1}':f for i,f in enumerate(fold_f1)},**pooled})
    table=pd.DataFrame(rows)
    # Prefer at least 20% recall for both directions in pooled development CV, if feasible.
    table['Recall guardrail']=table[['Down recall','Up recall']].min(axis=1)>=.20
    return table.sort_values(['Recall guardrail','Mean CV macro F1'],ascending=False,kind='stable').reset_index(drop=True)


def risk_metrics(y,p,threshold):
    y=np.asarray(y)!=0; pred=np.asarray(p)>=threshold
    return {'AP':average_precision_score(y,p),'F1':f1_score(y,pred,zero_division=0),
            'Precision':precision_score(y,pred,zero_division=0),'Recall':recall_score(y,pred,zero_division=0),
            'Accuracy':accuracy_score(y,pred),'Detected':int((y&pred).sum()),'False alarms':int((~y&pred).sum()),'Events':int(y.sum())}


def search_risk(X,y,training,feature_groups):
    folds=list(TimeSeriesSplit(n_splits=4,test_size=400,gap=5).split(X))
    for tr,va in folds:
        assert training.Outcome_date.iloc[tr].max()<training.Date.iloc[va].min()
    rows=[]
    for group,cols in feature_groups.items():
        for kind in ('logistic','forest','extra','hist'):
            for weighted in (False,True):
                blocks=[]
                for tr,va in folds:
                    model=estimator(kind,weighted).fit(X[cols].iloc[tr],y.iloc[tr].ne(0).astype(int))
                    blocks.append((y.iloc[va],model.predict_proba(X[cols].iloc[va])[:,1]))
                ap=np.mean([average_precision_score(a.ne(0),p) for a,p in blocks])
                # Pick the warning threshold by fold F1, after imposing pooled recall >= 30% if feasible.
                options=[]
                for threshold in np.arange(.05,.91,.05):
                    m=risk_metrics(pd.concat([a for a,_ in blocks]),np.concatenate([p for _,p in blocks]),threshold)
                    options.append({'threshold':float(threshold),'Mean CV F1':np.mean([risk_metrics(a,p,threshold)['F1'] for a,p in blocks]),**m})
                best=sorted(options,key=lambda q:(q['Recall']>=.30,q['Mean CV F1']),reverse=True)[0]
                rows.append({'features':group,'algorithm':kind,'weighted':weighted,'Mean CV AP':ap,**best})
    return pd.DataFrame(rows).sort_values('Mean CV AP',ascending=False,kind='stable').reset_index(drop=True)
