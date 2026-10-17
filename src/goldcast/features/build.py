"""Point-in-time feature panel.

Every raw observation is moved to its availability date
(`obs_date + lag_bdays` business days) before anything else happens, and each
decision date sees only values available on or before it (an as-of join).
`build_panel(..., cutoff=c)` must therefore give the same rows as the full
build on every decision date both contain; `tests/test_no_leakage.py` checks it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from goldcast.config import FreqSpec, SeriesSpec


def _available(s: pd.Series, spec: SeriesSpec, cutoff: pd.Timestamp | None) -> pd.Series:
    """Re-index a raw series by availability date, truncated at the cutoff."""
    s = s.dropna().sort_index()
    if spec.smooth_days:
        s = s.rolling(spec.smooth_days, min_periods=1).mean()
    if spec.lag_bdays:
        s.index = s.index + BDay(spec.lag_bdays)
    s = s[~s.index.duplicated(keep="last")]
    if cutoff is not None:
        s = s[s.index <= cutoff]
    return s


def _asof(s: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Last value available on or before each date."""
    if s.empty:
        return pd.Series(np.nan, index=dates)
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def _safe_log(v: pd.Series) -> pd.Series:
    return np.log(v.where(v > 0))


def decision_dates(target: pd.Series, freq: FreqSpec, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Last target trading day of each period, dropping a period that has not closed by `end`."""
    days = target.index.to_series()
    last = days.groupby(pd.Grouper(freq=freq.rule)).max().dropna()
    last = last[last.index <= end]  # the period label is its calendar end
    return pd.DatetimeIndex(last.values, name="date")


def build_panel(
    raw: dict[str, pd.Series],
    catalog: tuple[SeriesSpec, ...],
    freq: FreqSpec,
    target: str,
    cutoff: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) on decision dates. y_t = log(P_{t+1} / P_t); NaN when P_{t+1} is unknown."""
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else None
    specs = {s.name: s for s in catalog}
    tgt = _available(raw[target], specs[target], cutoff)
    end = cutoff if cutoff is not None else tgt.index.max()
    dates = decision_dates(tgt, freq, end)

    cols: dict[str, pd.Series] = {}
    for name, spec in specs.items():
        if name not in raw:
            continue
        s = _available(raw[name], spec, cutoff)
        v = _asof(s, dates)
        for kind in spec.kinds:
            if kind == "level":
                cols[f"{name}_level"] = v
            elif kind == "logret":
                lv = _safe_log(v)
                for w in freq.windows:
                    cols[f"{name}_logret{w}"] = lv - lv.shift(w)
            elif kind == "diff":
                for w in freq.windows:
                    cols[f"{name}_diff{w}"] = v - v.shift(w)
            elif kind == "rvol":
                daily = _safe_log(s).diff()
                for d in spec.rvol_days:
                    rv = daily.rolling(d, min_periods=max(5, d // 2)).std() * np.sqrt(252)
                    cols[f"{name}_rvol{d}d"] = _asof(rv, dates)
            else:
                raise ValueError(f"unknown feature kind {kind!r} for {name}")

    X = pd.DataFrame(cols, index=dates)
    price = _asof(tgt, dates)
    lp = _safe_log(price)
    y = (lp.shift(-1) - lp).rename("y")
    return X, y
