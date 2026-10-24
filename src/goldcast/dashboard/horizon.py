"""One forecast per horizon (1 week .. 2 years): live forecast, backtest board and track record.

1 week and 1 month reuse the R-002..R-005 cached backtests and refit the cheap models live
(`data.next_period`). 3 months to 2 years use a small month-end engine: random walk, drift and
the R-011 valuation ridge for the point, historical quantiles and trailing volatility for the
80% range. Every forecast uses only data known at its origin. "Best" means the lowest backtest
loss on the 2015+ test origins; whether that lead is significant is reported beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from goldcast.dashboard import data as D
from goldcast.evaluation import metrics as M

QS = D.QS
Z90 = norm.ppf(0.9)

# label -> (months, research frequency or None)
HORIZONS = {
    "1 week": (None, "weekly"),
    "1 month": (1, "monthly"),
    "3 months": (3, None),
    "6 months": (6, None),
    "1 year": (12, None),
    "2 years": (24, None),
}

NAMES = {
    "rw": "No change (random walk)",
    "drift": "Trend (gold's average past return)",
    "ses": "Smoothed recent returns",
    "ridge": "Ridge regression on 13 markets",
    "lgbm": "LightGBM on 13 markets",
    "valuation": "Valuation (real rates, dollar, price vs 5y average)",
    "chronos": "Chronos-2 (AI time-series model)",
    "chronos2x": "Chronos-2 with 12 market inputs",
    "tabpfn": "TabPFN-2 (AI tabular model)",
    "tabfm": "TabFM (AI tabular model)",
    "garch": "GARCH volatility",
    "ewma": "EWMA volatility",
    "qbase": "Historical quantiles (rolling)",
    "lgbm_q": "LightGBM quantiles",
    "hist": "Historical quantiles",
    "vol": "Trailing 1-year volatility",
}


def name(m: str) -> str:
    return NAMES.get(m, m)


@dataclass
class Horizon:
    label: str
    origin: pd.Timestamp       # last price date used
    target: pd.Timestamp       # date the forecast is for
    price: float               # gold at the origin
    point: dict                # model -> live log return
    quant: dict                # model -> live (q10, q50, q90) log returns
    point_board: pd.DataFrame  # backtest, ranked
    range_board: pd.DataFrame
    y: pd.Series               # backtest realized log returns by origin
    hist_point: pd.DataFrame   # backtest point forecasts by origin
    hist_quant: dict           # model -> DataFrame[QS] by origin
    step: pd.DateOffset        # origin -> target
    note: str

    @property
    def best_point(self) -> str:
        return next(m for m in self.point_board["model"] if m in self.point)

    @property
    def best_range(self) -> str:
        return next(m for m in self.range_board["model"] if m in self.quant)

    def p_up(self, rng: str, centre: float) -> float:
        """P(price above today) from the range's width, centred on the chosen point forecast."""
        q = self.quant[rng]
        sd = (q[2] - q[0]) / (2 * Z90)
        return float(norm.cdf(centre / sd)) if sd > 0 else float(centre > 0)


# ---------------------------------------------------------------- 1 week / 1 month
def short(cfg, label: str) -> Horizon:
    freq = HORIZONS[label][1]
    fc = D.load(freq)
    mask = np.ones(len(fc.y), bool)
    pb = D.point_board(fc, mask, cfg.regime_start)
    rb = D.interval_board(fc, mask, cfg.regime_start)
    live = D.next_period(cfg, freq)
    step = pd.DateOffset(weeks=1) if freq == "weekly" else pd.DateOffset(months=1)
    origin = live["data_last"]  # the monthly panel's inputs end at the last full month; applied to today's price
    from goldcast.data.store import load_raw
    price = float(load_raw(cfg)[cfg.target].dropna().iloc[-1])
    return Horizon(label, origin, origin + step, price, live["point"],
                   {m: tuple(q) for m, q in live["quant"].items()},
                   _friendly_points(pb), _friendly_ranges(rb, "pinball/garch"),
                   fc.y, fc.point, fc.quant, step,
                   "Backtest: R-002 to R-005 cached forecasts, one per "
                   f"{freq[:-2]} from 2015. AI models have backtest scores only (no live refit).")


# ---------------------------------------------------------------- 3 months .. 2 years
def _monthly(raw):
    g = raw["gold"].dropna().sort_index()
    me = lambda s: s.sort_index().shift(1).resample("ME").last()  # 1-bday lag, as R-011
    lgl = np.log(me(g))
    X = pd.DataFrame({"real10": me(raw["real10"]),
                      "d_real10": me(raw["real10"]).diff(12),
                      "d_dxy": np.log(me(raw["dxy"])).diff(12),
                      "gold_dev": lgl - lgl.rolling(60).mean()})
    lg = np.log(g.resample("ME").last())
    if g.index.max() < lg.index[-1] - pd.offsets.BDay(1):  # current month not finished
        lg = lg.iloc[:-1]
    r = np.log(g).diff()
    vol = r.rolling(252).std().shift(1).resample("ME").last()  # daily vol known at the month-end
    return g, lg, X, vol


def _fit_at(Xtr, ytr, xt, vol_t, h):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    drift = float(ytr.mean())
    pt = {"rw": 0.0, "drift": drift}
    if xt is not None and np.isfinite(xt).all():
        sc = StandardScaler().fit(Xtr)
        pt["valuation"] = float(Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr).predict(sc.transform(xt))[0])
    q = {"hist": tuple(np.quantile(ytr, QS))}
    if np.isfinite(vol_t):
        s = vol_t * np.sqrt(21 * h)
        q["vol"] = (drift - Z90 * s, drift, drift + Z90 * s)
    return pt, q


def long(cfg, raw, label: str) -> Horizon:
    h = HORIZONS[label][0]
    g, lg, X, vol = _monthly(raw)
    y = (lg.shift(-h) - lg).reindex(X.index)
    Dm = X.assign(y=y, vol=vol.reindex(X.index)).loc["2003-01":]
    cols = list(X.columns)
    rows = Dm.index[(Dm.index >= pd.Timestamp(cfg.test_start)) & Dm["y"].notna()]
    P, Q = {}, {}
    for t in rows:
        i = Dm.index.get_loc(t)
        tr = Dm.iloc[: i - h + 1].dropna(subset=cols + ["y"])
        if len(tr) < 24:
            continue
        xt = Dm.loc[[t], cols].values
        P[t], Q[t] = _fit_at(tr[cols].values, tr["y"].values, xt, Dm.loc[t, "vol"], h)
    hp = pd.DataFrame(P).T
    ys = Dm.loc[hp.index, "y"]
    hq = {m: pd.DataFrame({t: q[m] for t, q in Q.items() if m in q}).T.set_axis(list(QS), axis=1)
          for m in ("hist", "vol")}

    # live: everything with a known target trains; features at the last month-end row
    tr = Dm.dropna(subset=cols + ["y"])
    last = X.index[-1]
    pt, q = _fit_at(tr[cols].values, tr["y"].values, X.loc[[last], cols].values,
                    float(np.log(g).diff().iloc[-252:].std()), h)

    pb = _board_points(ys, hp, h, cfg.regime_start)
    rb = _board_ranges(ys, hq, h)
    origin = g.index.max()
    step = pd.DateOffset(months=h)
    return Horizon(label, origin, origin + step, float(g.iloc[-1]), pt, q,
                   _friendly_points(pb), _friendly_ranges(rb, "pinball/ref"),
                   ys, hp, hq, step,
                   f"Backtest: one forecast per month-end from 2015 ({len(ys)} origins). Targets overlap, "
                   f"so only about {max(1, len(ys) * 1 // h)} are independent; significance uses HAC lags.")


def _board_points(y, hp, h, regime_start):
    rw = y**2
    reg = y.index >= pd.Timestamp(regime_start)
    rows, ps = [], {}
    for m in hp:
        p = hp[m]
        ok = p.notna()
        se = (y - p) ** 2
        ps[m] = M.diebold_mariano(rw[ok].values, se[ok].values, h=h)[1] if m != "rw" else np.nan
        rows.append({"model": m, "n": int(ok.sum()), "mse/rw": se[ok].mean() / rw[ok].mean(),
                     "mse/rw 2023+": se[ok & reg].mean() / rw[ok & reg].mean() if (ok & reg).any() else np.nan,
                     "hit": float(((p[ok] > 0) == (y[ok] > 0)).mean())})
    out = pd.DataFrame(rows)
    out["p holm"] = out["model"].map(M.holm(ps))
    return D._rank(out, "mse/rw")


def _board_ranges(y, hq, h):
    ref = min(hq)  # "hist": scores are relative to historical quantiles
    base = M.pinball(y.values, np.sort(hq[ref].reindex(y.index).values, axis=1), QS)
    rows, ps = [], {}
    for m, q in hq.items():
        q = np.sort(q.reindex(y.index).values, axis=1)
        ok = np.isfinite(q).all(axis=1)
        pb = M.pinball(y.values[ok], q[ok], QS)
        ps[m] = M.diebold_mariano(base[ok], pb, h=h)[1] if m != ref else np.nan
        cov = M.coverage(y.values[ok], q[ok, 0], q[ok, 2])
        rows.append({"model": m, "n": int(ok.sum()), "pinball/ref": pb.mean() / base[ok].mean(),
                     "coverage 80%": cov, "coverage ok": 0.75 <= cov <= 0.85})
    out = pd.DataFrame(rows)
    out["p holm"] = out["model"].map(M.holm(ps))
    return D._rank(out, "pinball/ref")


def _friendly_points(b):
    return b[["rank", "model", "mse/rw", "mse/rw 2023+", "hit", "p holm", "significant"]].copy()


def _friendly_ranges(b, col):
    return b[["rank", "model", col, "coverage 80%", "coverage ok", "p holm", "significant"]].rename(
        columns={col: "loss/ref"}).copy()


def path(hz: Horizon, centre: float, q: tuple, n: int = 24) -> pd.DataFrame:
    """Price cone from the origin to the target: centre grows linearly, the band with sqrt(time)."""
    s = np.linspace(0, 1, n)
    dates = pd.DatetimeIndex([hz.origin + (hz.target - hz.origin) * x for x in s])
    lo, hi = q[0] - q[1], q[2] - q[1]
    return pd.DataFrame({"mid": hz.price * np.exp(centre * s),
                         "lo": hz.price * np.exp(centre * s + lo * np.sqrt(s)),
                         "hi": hz.price * np.exp(centre * s + hi * np.sqrt(s))}, index=dates)
