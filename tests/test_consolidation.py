import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.consolidation import predict_frame, FEATURES, BASE


def test_prediction_contract():
    rng=np.random.default_rng(42)
    x=pd.DataFrame(rng.normal(size=(30,11)),columns=FEATURES['all_four'])
    x['Date']=pd.date_range('2026-02-01',periods=28).tolist()+[pd.Timestamp('2026-03-01'),pd.Timestamp('2026-03-02')]
    x['Flow_month']=(x.Date.dt.to_period('M')-2).astype(str)
    y=np.tile([-1,0,1],10)
    models={name:RandomForestClassifier(n_estimators=3,random_state=42).fit(x[cols],y) for name,cols in FEATURES.items()}
    bundle={'models':models,'features':FEATURES,'risk_model':LogisticRegression().fit(x[BASE],y!=0),'risk_threshold':.2,'flow_months':['2025-12']}
    frame=x.iloc[:28].copy()
    for name in FEATURES:
        p=predict_frame(bundle,frame,name)
        assert p.Prediction.isin([-1,0,1]).all()
        np.testing.assert_allclose(p[['Score_Down','Score_Ordinary','Score_Up']].sum(axis=1),1)
        pd.testing.assert_frame_equal(p,predict_frame(bundle,frame.assign(Target=1,Forward_return_5d_pct=99),name))
    with pytest.raises(ValueError):
        predict_frame(bundle,frame.drop(columns=['FII_monthly_net']),'fii')
    with pytest.raises(ValueError):
        predict_frame(bundle,frame.assign(Flow_month='2026-02'),'all_four')
    with pytest.raises(ValueError):
        predict_frame(bundle,frame.assign(Return_1d=np.nan))
    with pytest.raises(ValueError):
        predict_frame(bundle,pd.concat([frame,frame]))
    with pytest.raises(ValueError):
        predict_frame(bundle,x.iloc[-2:],'fii')
