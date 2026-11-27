"""Preliminary monthly institutional flow comparison."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix, average_precision_score
from sklearn.model_selection import TimeSeriesSplit

MONTHLY_COLS = {g: [f'{g}_monthly_net', f'{g}_three_month_net'] for g in ('FII','DII')}

def monthly_features(monthly, dates):
    m=monthly[['Month','FII_net','DII_net']].copy()
    m.index=pd.PeriodIndex(m.pop('Month'),freq='M')
    m=m.sort_index().astype(float)
    assert m.index.is_unique and m.index.equals(pd.period_range(m.index.min(),m.index.max(),freq='M'))
    assert np.isfinite(m.to_numpy()).all()
    z=pd.DataFrame(index=m.index)
    for g in ('FII','DII'):
        z[MONTHLY_COLS[g][0]]=m[g+'_net']
        z[MONTHLY_COLS[g][1]]=m[g+'_net'].rolling(3).sum()
    # Use M-2 for forecast month M. Exact match, no indefinite carry-forward.
    d=pd.DatetimeIndex(pd.to_datetime(dates),name='Date')
    keys=d.to_period('M')-2
    out=z.reindex(keys).set_axis(d)
    out['Flow_month']=keys.astype(str)
    return out


def monthly_comparison(datasets, features, original_features):
    cols=MONTHLY_COLS['FII']+MONTHLY_COLS['DII']
    matched={}; audit=[]
    for period,part in datasets.items():
        part=part.copy(); part['Date']=pd.to_datetime(part.Date); part['Outcome_date']=pd.to_datetime(part.Outcome_date)
        assert part.Date.is_unique and part.Date.is_monotonic_increasing
        frame=part.join(features,on='Date',validate='one_to_one').dropna(subset=cols).reset_index(drop=True)
        matched[period]=frame
        audit.append({'Period':period,'Original rows':len(part),'Matched rows':len(frame),'Start':frame.Date.min(),'End':frame.Date.max(),'Major Down':int(frame.Target.eq(-1).sum()),'Ordinary':int(frame.Target.eq(0).sum()),'Major Up':int(frame.Target.eq(1).sum())})
    train=matched['train']
    folds=list(TimeSeriesSplit(n_splits=4,test_size=400,gap=5).split(train))
    for tr,va in folds:
        assert train.Outcome_date.iloc[tr].max()<train.Date.iloc[va].min()
        assert set(train.Target.iloc[tr])==set(train.Target.iloc[va])=={-1,0,1}
    def build():
        return RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,criterion='gini',max_features='sqrt',bootstrap=True,class_weight='balanced',random_state=42,n_jobs=1)
    rows=[]; preds=[]; fold_rows=[]
    for name,extra in [('Original',[]),('FII only',MONTHLY_COLS['FII']),('DII only',MONTHLY_COLS['DII']),('FII+DII',cols)]:
        c=list(original_features)+extra
        cv=[]
        for k,(tr,va) in enumerate(folds,1):
            model=build().fit(train[c].iloc[tr],train.Target.iloc[tr]); pred=model.predict(train[c].iloc[va]); score=f1_score(train.Target.iloc[va],pred,labels=[-1,0,1],average='macro',zero_division=0)
            cv.append(score); fold_rows.append({'Model':name,'Fold':k,'Macro F1':score})
        model=build().fit(train[c],train.Target)
        for period in ('validation','test'):
            p=matched[period]; assert train.Outcome_date.max()<p.Date.min(); y=p.Target; pred=model.predict(p[c]); prob=model.predict_proba(p[c]); assert model.classes_.tolist()==[-1,0,1]
            rec=recall_score(y,pred,labels=[-1,0,1],average=None,zero_division=0); matrix=confusion_matrix(y,pred,labels=[-1,0,1])
            rows.append({'Model':name,'Period':period,'Rows':len(y),'Mean CV macro F1':np.mean(cv),'Macro F1':f1_score(y,pred,labels=[-1,0,1],average='macro',zero_division=0),'Accuracy':accuracy_score(y,pred),'Balanced accuracy':balanced_accuracy_score(y,pred),'Down recall':rec[0],'Up recall':rec[2],'Major correct':int(((y!=0)&(y==pred)).sum()),'Ordinary false alarms':int(((y==0)&(pred!=0)).sum()),'Event AP':average_precision_score(y.ne(0),prob[:,0]+prob[:,2]),'Confusion matrix':str(matrix.tolist())})
            records=p[['Date','Outcome_date','Target','Flow_month']].copy(); records['Model']=name; records['Period']=period; records['Prediction']=pred; preds.append(records)
    return pd.DataFrame(rows),pd.DataFrame(audit),pd.concat(preds,ignore_index=True),pd.DataFrame(fold_rows)
