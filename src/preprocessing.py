"""Prepare the fixed NIFTY 50 and India VIX snapshots for KAIROS."""
from pathlib import Path

import numpy as np
import pandas as pd

PRICES = ["Open", "High", "Low", "Close"]
NIFTY_FILE = "KAIROS_NIFTY50_NSE_2006_2026.csv"
VIX_FILE = "KAIROS_INDIA_VIX_2008_2026.csv"


def prepare_data(nifty, vix):
    """Keep the shared date range and preserve the NIFTY trading calendar."""
    nifty, vix = nifty.copy(), vix.copy()
    assert set(nifty.columns) == set(["Date"] + PRICES)
    assert set(vix.columns) == {"Date", "VIX_Close"}

    for frame in (nifty, vix):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        assert frame.notna().all().all()
        assert frame.Date.is_unique
        assert frame.Date.is_monotonic_increasing

    assert np.isfinite(nifty[PRICES]).all().all()
    assert (nifty[PRICES] > 0).all().all()
    assert (nifty.High >= nifty[["Open", "Close", "Low"]].max(axis=1)).all()
    assert (nifty.Low <= nifty[["Open", "Close", "High"]].min(axis=1)).all()
    assert np.isfinite(vix.VIX_Close).all() and (vix.VIX_Close > 0).all()

    start = max(nifty.Date.min(), vix.Date.min())
    end = min(nifty.Date.max(), vix.Date.max())
    assert start <= end
    aligned = nifty.loc[nifty.Date.between(start, end)].copy()
    vix_aligned = vix.loc[vix.Date.between(start, end)].copy()
    assert vix_aligned.Date.isin(aligned.Date).all()

    df = aligned.merge(vix_aligned, on="Date", how="left", validate="one_to_one")
    df["VIX_missing"] = df.VIX_Close.isna().astype(int)
    # Use past observations only; retain a flag for the imputed value.
    df["VIX_Close"] = df.VIX_Close.ffill()
    assert df.notna().all().all()
    assert len(df) == len(aligned)
    assert df.Date.is_unique and df.Date.is_monotonic_increasing
    return df


def load_and_prepare(data_dir):
    data_dir = Path(data_dir)
    return prepare_data(pd.read_csv(data_dir / NIFTY_FILE),
                        pd.read_csv(data_dir / VIX_FILE))
