import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from src.final_evaluation import prediction_records


def test_records_alignment_and_probabilities():
    data = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=6),
                         'Outcome_date': pd.date_range('2020-01-06', periods=6),
                         'Forward_return_5d_pct': [-4, 0, 4, -5, 1, 5],
                         'Target': [-1, 0, 1, -1, 0, 1], 'a': range(6)})
    model = DummyClassifier(strategy='prior').fit(data[['a']], data.Target)
    records = prediction_records(model, data, ['a'])
    assert records.index.equals(data.index)
    np.testing.assert_allclose(records[['Score_Down', 'Score_Ordinary', 'Score_Up']].sum(axis=1), 1)
    assert records.Correct.equals(records.Prediction.eq(records.Target))
    assert records.Target.equals(data.Target)
