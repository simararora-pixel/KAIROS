import numpy as np
import pandas as pd
import pytest
from src.reference import FEATURES, build_reference, predict_reference


def test_single_model_prediction_contract():
    x = pd.DataFrame(np.random.default_rng(42).normal(size=(90,7)), columns=FEATURES)
    x['Date'] = pd.date_range('2020-01-01', periods=len(x))
    model = build_reference().fit(x[FEATURES], np.tile([-1,0,1],30))
    p = predict_reference(model,x)
    assert model.n_features_in_ == 7
    assert model.get_params()['max_depth'] == 8
    np.testing.assert_allclose(p[['Score_Down','Score_Ordinary','Score_Up']].sum(axis=1),1)
    pd.testing.assert_frame_equal(p,predict_reference(model,x.assign(Target=1,Forward_return_5d_pct=99)))
    with pytest.raises(ValueError):
        predict_reference(model,x.assign(Return_1d=np.nan))
    with pytest.raises(ValueError):
        predict_reference(model,x.drop(columns=['VIX_Close']))
    with pytest.raises(ValueError):
        predict_reference(model,pd.concat([x,x]))
