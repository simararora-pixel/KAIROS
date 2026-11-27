"""Independent snapshot arithmetic and inference reconciliation. Run from project root."""
from pathlib import Path
import numpy as np
import pandas as pd
from dashboard.core import load_workspace, FEATURES, SCORES, evaluation


def audit(root):
    w=load_workspace(root)
    n,v=w['nifty'],w['vix']; d=w['market']; f=w['dataset']; model=w['bundle']['model']
    assert n.Date.is_unique and v.Date.is_unique
    p=['Open','High','Low','Close']
    assert np.isfinite(n[p]).all().all() and (n[p]>0).all().all()
    assert (n.High>=n[p].max(axis=1)).all() and (n.Low<=n[p].min(axis=1)).all()
    assert np.isfinite(v.VIX_Close).all() and (v.VIX_Close>0).all()
    a=n.loc[n.Date.between(v.Date.min(),v.Date.max())].merge(v,on='Date',how='left',validate='one_to_one').reset_index(drop=True)
    missing=np.flatnonzero(a.VIX_Close.isna())
    assert a.Date.iloc[missing].tolist()==w['imputation'].Date.tolist()==['2024-03-02']
    a.VIX_Close=a.VIX_Close.ffill()
    pd.testing.assert_frame_equal(a,d,check_dtype=False,atol=1e-10)
    c=d.Close.to_numpy(); vx=d.VIX_Close.to_numpy(); dates=d.Date.tolist(); pos={x:i for i,x in enumerate(dates)}
    bounds={'train':(0,int(len(d)*.7)),'validation':(int(len(d)*.7),int(len(d)*.85)),'test':(int(len(d)*.85),len(d))}
    expected=[]; bad=set(missing)|{i+1 for i in missing}
    for name,(lo,hi) in bounds.items():
        part=f.loc[f.Partition.eq(name)]
        ix=[i for i in range(max(lo,20),hi-5) if i not in bad]
        assert part.Date.tolist()==[dates[i] for i in ix]
        expected.extend(ix)
        for r,i in zip(part.itertuples(index=False),ix):
            returns=(c[i-19:i+1]/c[i-20:i]-1)*100
            actual=[(c[i]/c[i-1]-1)*100,(c[i]/c[i-5]-1)*100,np.std(returns,ddof=1),(c[i]/np.mean(c[i-19:i+1])-1)*100,(d.High.iloc[i]-d.Low.iloc[i])/d.Open.iloc[i]*100,vx[i],(vx[i]/vx[i-1]-1)*100]
            np.testing.assert_allclose([getattr(r,k) for k in FEATURES],actual,atol=1e-10,rtol=0)
            ret=(c[i+5]/c[i]-1)*100
            label=0 if abs(ret)<=3 or np.isclose(abs(ret),3,atol=1e-10,rtol=0) else int(np.sign(ret))
            assert r.Outcome_date==dates[i+5] and r.Target==label
            np.testing.assert_allclose(r.Forward_return_5d_pct,ret,atol=1e-10,rtol=0)
        if name!='train':
            saved=w[name]; assert saved.Date.tolist()==part.Date.tolist()
            assert saved.Target.tolist()==part.Target.tolist() and saved.Outcome_date.tolist()==part.Outcome_date.tolist()
            np.testing.assert_allclose(saved.Forward_return_5d_pct,part.Forward_return_5d_pct,atol=1e-10)
            np.testing.assert_array_equal(model.predict(part[FEATURES]),saved.Prediction)
            np.testing.assert_allclose(model.predict_proba(part[FEATURES]),saved[SCORES],atol=1e-12)
            evaluation(w,name)
    assert len(expected)==len(f)==4013 and len(set(expected))==len(f)
    assert w['inputs'].Date.tolist()==dates[-5:]
    np.testing.assert_array_equal(model.predict(w['inputs'][FEATURES]),w['latest'].Prediction)
    print('PASS: raw join, all 4013 eligible rows x 7 features, all targets and outcome dates, both evaluation partitions, latest predictions')
    print('Weekend observations:', int((pd.to_datetime(d.Date).dt.dayofweek>=5).sum()))
    print('NOTE: internal consistency is not full external source reconciliation.')

if __name__=='__main__':
    audit(Path(__file__).resolve().parents[1])
