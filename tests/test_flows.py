import numpy as np
import pandas as pd
import pytest
from src.flows import prepare_flow


def test_previous_session_and_missing_history():
    dates=pd.bdate_range('2020-01-01',periods=40)
    frame=pd.DataFrame({'Date':dates,'Buy':100+np.arange(40),'Sell':100,'Net':np.arange(40)})
    out=prepare_flow(frame,dates,'FII')
    assert out.iloc[21].FII_net_lag1 == 20
    assert out.iloc[:20].FII_imbalance_20d.isna().all()
    modified=frame.copy(); modified.loc[21,['Buy','Net']]=[200,100]
    pd.testing.assert_series_equal(out.iloc[21],prepare_flow(modified,dates,'FII').iloc[21])
    pd.testing.assert_frame_equal(out.iloc[:25],prepare_flow(frame.iloc[:25],dates[:25],'FII'))
    missing=prepare_flow(frame.drop(index=20),dates,'FII')
    assert pd.isna(missing.iloc[21].FII_net_lag1)
    assert pd.isna(missing.iloc[25].FII_imbalance_20d)
    with pytest.raises(ValueError):
        prepare_flow(pd.concat([frame,frame.iloc[:1]]),dates,'FII')
    with pytest.raises(ValueError):
        prepare_flow(frame.assign(Net=100),dates,'DII')
