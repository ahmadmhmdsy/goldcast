"""Read every study's cached forecasts into one frame per frequency and score them.

Only the report CSVs are read; nothing is refit. Scores here are recomputed on whatever
window the dashboard selects, so they match the pre-registered reports only on the full
test window, and the Holm column is over the models shown, not a study's frozen family.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from goldcast.config import ROOT
from goldcast.evaluation import metrics as M

REPORTS = ROOT / "reports"
QS = (0.1, 0.5, 0.9)

# model -> (study, short description)
MODELS = {
    "rw": ("R-002", "random walk: forecasts a zero return"),
    "drift": ("R-002", "historical mean return"),
    "ses": ("R-002", "simple exponential smoothing"),
    "ridge": ("R-002", "ridge on the 13-series panel"),
    "lgbm": ("R-002", "LightGBM on the 13-series panel"),
    "qbase": ("R-002", "empirical quantiles of past returns"),
    "lgbm_q": ("R-002", "LightGBM quantile regression"),
    "chronos": ("R-002", "Chronos-2 on gold returns only"),
    "ewma": ("R-003", "EWMA vol with filtered historical simulation"),
    "garch": ("R-003", "GARCH(1,1) with t errors"),
    "lgbm+B1": ("R-004", "lgbm plus flows/positioning"),
    "lgbm+B2": ("R-004", "lgbm plus China-India FX"),
    "lgbm+B3": ("R-004", "lgbm plus uncertainty indices"),
    "lgbm+B4": ("R-004", "lgbm plus Google Trends attention"),
    "lgbm+ALL": ("R-004", "lgbm plus every block"),
    "tabpfn": ("R-005", "TabPFN-2 on the panel"),
    "tabfm": ("R-005", "TabFM v1.0.0 on the panel (point only)"),
    "chronos2x": ("R-005", "Chronos-2 with 12 past covariates"),
}


@dataclass
class Forecasts:
    y: pd.Series
    point: pd.DataFrame  # one column per model
    quant: dict[str, pd.DataFrame] = field(default_factory=dict)  # model -> columns QS


def _csv(path, **kw):
    return pd.read_csv(path, index_col=0, parse_dates=[0], **kw) if path.exists() else None


def load(freq: str) -> Forecasts:
    base = _csv(REPORTS / freq / "predictions.csv")
    y = base["y"]
    point = pd.DataFrame({c.split(":", 1)[1]: base[c] for c in base if c.startswith("point:")})
    quant = {}

    iv = REPORTS / "intervals" / f"{freq}_predictions.csv"
    if iv.exists():
        t = pd.read_csv(iv, header=[0, 1], index_col=0, skiprows=[2])
        t.index = pd.to_datetime(t.index)
        for m in dict.fromkeys(c[0] for c in t.columns if c[0] != "y"):
            quant[m] = t[m].set_axis(list(QS), axis=1)
        if "chronos" in quant:
            point["chronos"] = quant["chronos"][0.5]

    bl = _csv(REPORTS / "blocks" / f"{freq}_predictions.csv")
    if bl is not None:
        for c in bl:
            b, m = c.split(":")
            if m == "lgbm" and b != "base":
                point[f"lgbm+{b}"] = bl[c]

    for arm in ("tabpfn", "tabfm", "chronos2x"):
        f = _csv(REPORTS / "foundation" / f"{freq}_{arm}.csv")
        if f is None:
            continue
        point[arm] = f["point"]
        if "0.1" in f:
            quant[arm] = f[["0.1", "0.5", "0.9"]].set_axis(list(QS), axis=1)

    idx = y.dropna().index
    return Forecasts(y.loc[idx], point.reindex(idx), {k: v.reindex(idx) for k, v in quant.items()})


def point_board(fc: Forecasts, mask, regime_start: str) -> pd.DataFrame:
    y = fc.y[mask].values
    reg = fc.y.index[mask] >= pd.Timestamp(regime_start)
    rw = y**2
    rows, ps = [], {}
    for m in fc.point:
        p = fc.point[m][mask].values
        ok = np.isfinite(p)
        if ok.sum() < 10:
            continue
        se = (y - p) ** 2
        _, pv = M.diebold_mariano(rw[ok], se[ok]) if m != "rw" else (np.nan, np.nan)
        ps[m] = pv
        rows.append({"model": m, "study": MODELS.get(m, ("?",))[0], "n": int(ok.sum()),
                     "mse/rw": se[ok].mean() / rw[ok].mean(),
                     "mse/rw 2023+": se[ok & reg].mean() / rw[ok & reg].mean() if (ok & reg).any() else np.nan,
                     "hit": float(((p[ok] > 0) == (y[ok] > 0)).mean()), "p vs rw": pv})
    out = pd.DataFrame(rows)
    out["p holm"] = out["model"].map(M.holm(ps))
    return _rank(out, "mse/rw")


def interval_board(fc: Forecasts, mask, regime_start: str) -> pd.DataFrame:
    y = fc.y[mask].values
    reg = fc.y.index[mask] >= pd.Timestamp(regime_start)
    g = M.pinball(y, np.sort(fc.quant["garch"][mask].values, axis=1), QS)
    rows, ps = [], {}
    for m, q in fc.quant.items():
        q = np.sort(q[mask].values, axis=1)
        ok = np.isfinite(q).all(axis=1)
        pb = M.pinball(y[ok], q[ok], QS)
        _, pv = M.diebold_mariano(g[ok], pb) if m != "garch" else (np.nan, np.nan)
        ps[m] = pv
        cov = M.coverage(y[ok], q[ok, 0], q[ok, 2])
        rows.append({"model": m, "study": MODELS.get(m, ("?",))[0], "n": int(ok.sum()),
                     "pinball/garch": pb.mean() / g[ok].mean(),
                     "pinball/garch 2023+": (pb[reg[ok]].mean() / g[ok & reg].mean()) if (ok & reg).any() else np.nan,
                     "coverage 80%": cov, "coverage ok": 0.75 <= cov <= 0.85, "p vs garch": pv})
    out = pd.DataFrame(rows)
    out["p holm"] = out["model"].map(M.holm(ps))
    return _rank(out, "pinball/garch")


def _rank(df: pd.DataFrame, by: str) -> pd.DataFrame:
    df = df.sort_values(by).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df["beats ref"] = df[by] < 1
    df["significant"] = df["p holm"] < 0.05
    return df


def cumulative_gain(fc: Forecasts, model: str, mask) -> pd.Series:
    """Running sum of (rw loss - model loss); rising means the model is ahead of the random walk."""
    y, p = fc.y[mask], fc.point[model][mask]
    d = y**2 - (y - p) ** 2
    return d.fillna(0).cumsum()


def cumulative_gain_interval(fc: Forecasts, model: str, mask) -> pd.Series:
    """Running sum of (garch pinball - model pinball)."""
    y = fc.y[mask].values
    g = M.pinball(y, np.sort(fc.quant["garch"][mask].values, axis=1), QS)
    q = np.sort(fc.quant[model][mask].values, axis=1)
    return pd.Series(np.nan_to_num(g - M.pinball(y, q, QS)), index=fc.y.index[mask]).cumsum()


def next_period(cfg, freq: str) -> dict:
    """Fit the cheap models on every known target and forecast the period after the latest row.

    Uses the stored raw data as fetched; foundation models are left out (minutes to hours per fit).
    """
    from goldcast.data.store import load_raw
    from goldcast.features.build import build_panel
    from goldcast.models import REGISTRY
    from goldcast.models.vol import EWMAQuantile, GARCHQuantile

    raw = load_raw(cfg)
    fs = cfg.freqs[freq]
    X, y = build_panel(raw, cfg.base_series, fs, cfg.target)
    known, last = y.notna(), X.iloc[[-1]]
    price = raw[cfg.target]
    point = {m: float(np.ravel(f(cfg, fs).fit(X[known], y[known]).predict(last))[0])
             for m, f in REGISTRY["point"].items()}
    qf = {**{m: (lambda f=f: f(cfg, fs)) for m, f in REGISTRY["quantile"].items()},
          "ewma": lambda: EWMAQuantile(cfg.quantiles), "garch": lambda: GARCHQuantile(cfg.quantiles)}
    quant = {m: np.sort(np.ravel(f().fit(X[known], y[known]).predict(last))) for m, f in qf.items()}
    # R-006/R-007: GVZ implied variance was the best forecast of move size (QLIKE), not of the range
    ndays = {"weekly": 5, "monthly": 21}.get(freq, 5)
    gvz = float(last["gvz_level"].iloc[0]) if "gvz_level" in last else np.nan
    return {"origin": last.index[0], "price": float(price.loc[: last.index[0]].iloc[-1]),
            "data_last": price.index.max(), "point": point, "quant": quant,
            "gvz": gvz, "gvz_move": gvz / 100 * np.sqrt(ndays / 252)}
