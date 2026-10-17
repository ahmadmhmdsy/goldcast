import numpy as np
import pandas as pd
import pytest

from goldcast.config import FreqSpec, SeriesSpec


@pytest.fixture
def synthetic():
    """Offline raw data shaped like the real sources: market closes, a lagged FRED-like rate,
    and a 7-day-a-week index published with a delay."""
    rng = np.random.default_rng(0)
    bd = pd.bdate_range("2005-01-03", "2012-12-31")
    cal = pd.date_range("2005-01-01", "2012-12-31")
    raw = {
        "gold": pd.Series(400 * np.exp(np.cumsum(rng.normal(0, 0.01, bd.size))), index=bd),
        "oil": pd.Series(50 + np.cumsum(rng.normal(0, 1, bd.size)), index=bd),  # can go negative
        "rate": pd.Series(2 + np.cumsum(rng.normal(0, 0.02, bd.size)), index=bd).iloc[::1],
        "gpr": pd.Series(100 + rng.normal(0, 10, cal.size), index=cal),
    }
    catalog = (
        SeriesSpec("gold", "x", "g", 0, ("logret", "rvol"), rvol_days=(20, 60)),
        SeriesSpec("oil", "x", "o", 0, ("logret",)),
        SeriesSpec("rate", "x", "r", 1, ("level", "diff")),
        SeriesSpec("gpr", "x", "p", 7, ("level", "diff"), smooth_days=7),
    )
    freqs = {
        "weekly": FreqSpec("weekly", "W-FRI", (1, 4, 13), 4, 52, 52),
        "monthly": FreqSpec("monthly", "ME", (1, 3, 6), 3, 24, 12),
    }
    return raw, catalog, freqs
