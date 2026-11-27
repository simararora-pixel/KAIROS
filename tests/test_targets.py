import pandas as pd
from src.targets import make_targets


def test_threshold_boundaries():
    for future, expected in [(96, -1), (97, 0), (100, 0), (103, 0), (104, 1)]:
        example = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=6), 'Close': [100]*5 + [future]})
        result = make_targets(example)
        assert result.Target.iloc[0] == expected
        assert result.Target.iloc[1:].isna().all()


def test_partition_endpoint():
    example = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=20), 'Close': list(range(100, 120))})
    for block in (example.iloc[:10], example.iloc[10:]):
        result = make_targets(block)
        assert result.Target.tail(5).isna().all()
        assert result.Outcome_date.dropna().le(block.Date.max()).all()
        assert result.Target.notna().sum() == len(block) - 5
