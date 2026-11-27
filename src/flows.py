"""Cash-market FII/DII features and matched retrospective evaluation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

FLOW_COLUMNS = [f'{g}_{s}' for g in ('FII','DII') for s in ('net_lag1','imbalance_lag1','imbalance_5d','imbalance_20d')]


def prepare_flow(frame, sessions, group):
    """Input: Date, Buy, Sell, Net (INR crore); align to previous market session."""
    if group not in ('FII', 'DII'):
        raise ValueError('Invalid investor category')
    f = frame[['Date','Buy','Sell','Net']].copy()
    f.Date = pd.to_datetime(f.Date, errors='raise').dt.normalize()
    if f.Date.isna().any() or f.Date.duplicated().any():
        raise ValueError('Missing or duplicate flow dates')
    for col in ('Buy','Sell','Net'):
        f[col] = pd.to_numeric(f[col].astype(str).str.replace(',','',regex=False), errors='raise')
    if not np.isfinite(f[['Buy','Sell','Net']]).all().all() or (f[['Buy','Sell']] < 0).any().any():
        raise ValueError('Invalid flow amounts')
    if not np.allclose(f.Buy-f.Sell, f.Net, atol=.05, rtol=0):
        raise ValueError('Net must equal Buy minus Sell (rounding tolerance 0.05 crore)')
    dates = pd.DatetimeIndex(pd.to_datetime(sessions), name='Date')
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Market sessions must be unique and chronological')
    f = f.set_index('Date').reindex(dates)
    # Never impute missing daily flows or use same-session post-close releases.
    gross = f.Buy + f.Sell
    imbalance = f.Net / gross.replace(0,np.nan)
    out = pd.DataFrame({f'{group}_net_lag1':f.Net.shift(1),
                        f'{group}_imbalance_lag1':imbalance.shift(1),
                        f'{group}_imbalance_5d':(f.Net.rolling(5).sum()/gross.rolling(5).sum().replace(0,np.nan)).shift(1),
                        f'{group}_imbalance_20d':(f.Net.rolling(20).sum()/gross.rolling(20).sum().replace(0,np.nan)).shift(1)}, index=dates)
    return out.replace([np.inf,-np.inf],np.nan)


def compare_flows(datasets, flow_features, original_features):
    """Same forecast dates and model settings; only inputs differ."""
    matched, audit = {}, []
    for name, part in datasets.items():
        merged = part.join(flow_features, on='Date', validate='one_to_one')
        matched[name] = merged.dropna(subset=FLOW_COLUMNS).reset_index(drop=True)
        audit.append({'Partition':name,'Original rows':len(part),'Matched rows':len(matched[name]),'Coverage':len(matched[name])/len(part)})
    audit = pd.DataFrame(audit)
    if (audit.Coverage < .8).any():
        raise ValueError('Insufficient coverage: '+audit.to_string(index=False))
    train = matched['train']
    folds = list(TimeSeriesSplit(n_splits=4,test_size=400,gap=5).split(train))
    for tr, va in folds:
        if train.Outcome_date.iloc[tr].max() >= train.Date.iloc[va].min():
            raise ValueError('Overlapping outcomes')
        if set(train.Target.iloc[tr]) != {-1,0,1} or set(train.Target.iloc[va]) != {-1,0,1}:
            raise ValueError('Missing class in fold')
    results, models = [], {}
    for name, cols in {'Original':list(original_features),'FII+DII':list(original_features)+FLOW_COLUMNS}.items():
        def build():
            return RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,criterion='gini',max_features='sqrt',bootstrap=True,class_weight='balanced',random_state=42,n_jobs=1)
        cv = [f1_score(train.Target.iloc[va],build().fit(train[cols].iloc[tr],train.Target.iloc[tr]).predict(train[cols].iloc[va]),labels=[-1,0,1],average='macro',zero_division=0) for tr,va in folds]
        model = build().fit(train[cols],train.Target)
        models[name] = model
        for period in ('validation','test'):
            part=matched[period]
            if train.Outcome_date.max() >= part.Date.min():
                raise ValueError('Training outcomes overlap evaluation')
            pred=model.predict(part[cols]); y=part.Target
            recalls=recall_score(y,pred,labels=[-1,0,1],average=None,zero_division=0)
            results.append({'Model':name,'Period':period,'Rows':len(y),'Mean CV macro F1':np.mean(cv),'Macro F1':f1_score(y,pred,labels=[-1,0,1],average='macro',zero_division=0),'Accuracy':accuracy_score(y,pred),'Balanced accuracy':balanced_accuracy_score(y,pred),'Down recall':recalls[0],'Up recall':recalls[2],'Major correct':int(((y!=0)&(y==pred)).sum()),'Ordinary false alarms':int(((y==0)&(pred!=0)).sum())})
    return pd.DataFrame(results), audit, models
