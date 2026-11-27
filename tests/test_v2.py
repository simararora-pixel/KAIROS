import numpy as np
import pandas as pd
from src.v2 import expanded_features, fit_scores, scores, decide, metrics, V1


def test_causal_features_and_two_stage():
    rng = np.random.default_rng(9)
    c = pd.Series(100 + np.cumsum(rng.normal(size=150)))
    df = pd.DataFrame({'Close':c,'Open':c-.1,'High':c+1,'Low':c-1,'VIX_Close':20.})
    base = pd.DataFrame(rng.normal(size=(150,7)), columns=V1)
    z = expanded_features(df,base)
    pd.testing.assert_frame_equal(z.iloc[:100], expanded_features(df.iloc[:100],base.iloc[:100]))
    assert z.shape[1] == 22
    y = pd.Series(np.tile([-1,0,1],50))
    for architecture in ('direct','two_stage'):
        model = fit_scores(base,y,architecture,'logistic',True)
        p = scores(model,base)
        np.testing.assert_allclose(p.sum(axis=1),1)
        assert set(decide(p,('gate',.3,.5))) <= {-1,0,1}
    assert decide(np.array([[.4,.4,.2],[.1,.8,.1],[.1,.3,.6]]),('gate',.5,.5)).tolist()==[-1,0,1]
    assert metrics([-1,0,1],[-1,0,1])['Macro F1']==1
