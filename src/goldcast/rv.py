"""R-006: next-period gold realized variance, HAR-RV reference vs GARCH, HAR-X (GVZ) and LightGBM.

Target for decision date D_t: the sum of squared daily gold log returns on days in
(D_t, D_t+1]. Every arm forecasts that variance; scoring is QLIKE. Training rows start
at the first GVZ value (2008-06) for all arms. `ndays` (trading days in the next period)
comes from the trading calendar, which is known ahead except for unscheduled closures.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from goldcast.blocks import _limit
from goldcast.config import Config
from goldcast.evaluation import metrics as M
from goldcast.evaluation.walkforward import walk_forward
from goldcast.features.build import build_panel
from goldcast.models.ml import LGBM_PARAMS

HAR = ["har_d", "har_w", "har_m"]
FLOOR = 1e-8
Z90 = 1.2815515655446004  # 80% central interval


def frame(cfg: Config, raw, fs, limit=None):
    """(F, rv): panel features plus HAR terms, log GVZ^2 and ndays; rv is the next-period target."""
    X, _ = build_panel(raw, cfg.base_series, fs, cfg.target)
    r = np.log(raw[cfg.target]).diff().dropna()
    r2 = r**2
    D = X.index
    nxt = pd.Series(D[1:].append(pd.DatetimeIndex([pd.NaT])), index=D)
    bucket = pd.Series(D.searchsorted(r.index, side="left"), index=r.index)  # day d -> first D >= d
    rv = r2.groupby(bucket.values).sum()
    nd = r2.groupby(bucket.values).size()
    # rows k of rv cover (D[k-1], D[k]]; row t's target covers (D[t], D[t+1]] = k = t + 1
    pos = np.arange(len(D)) + 1
    y = pd.Series(rv.reindex(pos).values, index=D, name="rv")
    y[nxt.isna().values] = np.nan
    F = X.copy()
    F["ndays"] = nd.reindex(pos).values
    F["ndays"] = F["ndays"].fillna(F["ndays"].median())  # last row: period not closed
    asof = lambda s: s.reindex(s.index.union(D)).ffill().reindex(D)  # noqa: E731
    F["har_d"] = asof(r2)
    F["har_w"] = asof(r2.rolling(5).mean())
    F["har_m"] = asof(r2.rolling(22).mean())
    F["log_gvz2"] = np.log(F["gvz_level"] ** 2)
    first = F["log_gvz2"].first_valid_index()
    F, y = F.loc[first:], y.loc[first:]
    F, y = _limit(cfg, F, y, limit)
    return F, y, r


class HARRV:
    cheap = True
    cols = HAR

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X[self.cols].values])
        ok = np.isfinite(A).all(axis=1) & np.isfinite(y.values)
        self.b_ = np.linalg.lstsq(A[ok], y.values[ok], rcond=None)[0]
        return self

    def predict(self, X):
        A = np.column_stack([np.ones(len(X)), X[self.cols].values])
        return np.maximum(A @ self.b_, FLOOR)


class HARX(HARRV):
    cols = HAR + ["log_gvz2"]


class LGBMLogRV:
    """LightGBM on log RV; back-transformed with Duan's smearing factor."""

    def fit(self, X, y):
        from lightgbm import LGBMRegressor

        ok = y.values > 0
        Z, t = self._z(X[ok]), np.log(y.values[ok])
        self.m_ = LGBMRegressor(**LGBM_PARAMS).fit(Z, t)
        self.smear_ = float(np.mean(np.exp(t - self.m_.predict(Z))))
        return self

    def predict(self, X):
        return np.maximum(np.exp(self.m_.predict(self._z(X))) * self.smear_, FLOOR)

    @staticmethod
    def _z(X):
        Z = X.drop(columns=["ndays"]).copy()
        for c in HAR:
            Z[c] = np.log(Z[c].clip(lower=FLOOR))
        return Z


class GARCHDaily:
    """GARCH(1,1)-t on daily returns up to the origin; the next ndays variances are summed."""

    cheap = True

    def __init__(self, r, start):
        self.r, self.start = r, start

    def fit(self, X, y):
        return self

    def predict(self, X):
        from arch import arch_model

        out = []
        for d, n in zip(X.index, X["ndays"].values):
            x = 100 * self.r.loc[self.start:d].values
            am = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", show_warning=False)
            v = res.forecast(horizon=int(n), reindex=False).variance.values[-1]
            out.append(v.sum() / 1e4)
        return np.array(out)


def qlike(rv, f):
    x = rv / f
    return x - np.log(x) - 1


def run(cfg: Config, raw, freqs, limit=None) -> tuple[dict, dict]:
    scores, preds, fam = {}, {}, {}
    for f in freqs:
        fs = cfg.freqs[f]
        F, y, r = frame(cfg, raw, fs, limit)
        start = F.index[0]
        P = {"har": walk_forward(HARRV, F, y, cfg.test_start, fs.refit_every),
             "harx": walk_forward(HARX, F, y, cfg.test_start, fs.refit_every),
             "lgbm": walk_forward(LGBMLogRV, F, y, cfg.test_start, fs.refit_every),
             "garch": walk_forward(lambda: GARCHDaily(r, start), F, y, cfg.test_start, fs.refit_every)}
        idx = P["har"].index
        P["gvz"] = pd.Series((F.loc[idx, "gvz_level"] / 100) ** 2 / 252 * F.loc[idx, "ndays"], index=idx)
        P = pd.DataFrame(P)
        rv = y.loc[idx].values
        # period return over the same window as rv, for interval coverage
        lp = np.log(raw[cfg.target])
        price = lp.reindex(lp.index.union(F.index)).ffill().reindex(F.index)
        ret = (price.shift(-1) - price).loc[idx].values
        reg = idx >= pd.Timestamp(cfg.regime_start)
        loss = {m: qlike(rv, P[m].values) for m in P}
        s = {"n_test": len(idx), "test_first": str(idx.min().date()), "test_last": str(idx.max().date()),
             "n_regime": int(reg.sum()), "train_start": str(start.date()), "models": {}}
        for m in P:
            sd = np.sqrt(P[m].values)
            s["models"][m] = {
                "qlike": float(loss[m].mean()), "qlike_regime": float(loss[m][reg].mean()),
                "qlike_ratio_vs_har": float(loss[m].mean() / loss["har"].mean()),
                "mse_logrv": float(np.mean((np.log(rv) - np.log(P[m].values)) ** 2)),
                "coverage": M.coverage(ret, -Z90 * sd, Z90 * sd),
                "coverage_regime": M.coverage(ret[reg], -Z90 * sd[reg], Z90 * sd[reg])}
        for v, (m, ref) in {"V1": ("garch", "har"), "V2": ("harx", "har"), "V3": ("lgbm", "harx")}.items():
            _, p = M.diebold_mariano(loss[ref], loss[m])
            s.setdefault("tests", {})[v] = {"model": m, "vs": ref, "p": p}
            fam[(f, v)] = p
        scores[f], preds[f] = s, P.assign(rv=rv, ret=ret)
    for (f, v), p in M.holm(fam).items():
        t = scores[f]["tests"][v]
        t["p_holm"] = p
        t["passes"] = bool(p < cfg.alpha)
    for f, s in scores.items():
        useful = []
        for v, t in s["tests"].items():
            m = s["models"][t["model"]]
            if t["passes"] and (v != "V3" or s["models"]["lgbm"]["qlike"] < s["models"]["har"]["qlike"]) \
                    and 0.75 <= m["coverage"] <= 0.85:
                useful.append(t["model"])
        s["useful_range_engine"] = useful
    return scores, preds


def to_markdown(scores: dict) -> str:
    v2 = [s["tests"]["V2"]["passes"] for s in scores.values()]
    v1 = [s["tests"]["V1"]["passes"] for s in scores.values()]
    v3 = [s["tests"]["V3"]["passes"] for s in scores.values()]
    lines = [f"# R-006 realized vol (Holm family {3 * len(scores)})", "",
             f"V1 GARCH beats HAR (prior: no): {v1}; V2 HAR-X beats HAR: {v2}; V3 LightGBM beats HAR-X: {v3}", ""]
    for f, s in scores.items():
        lines += [f"## {f} ({s['n_test']} origins, {s['test_first']}..{s['test_last']}; "
                  f"{s['n_regime']} in 2023+; training from {s['train_start']})", "",
                  "| model | QLIKE | QLIKE/har | QLIKE 2023+ | MSE log RV | cov 80% | cov 2023+ |",
                  "|---|---|---|---|---|---|---|"]
        for m, r in s["models"].items():
            lines.append(f"| {m} | {r['qlike']:.4f} | {r['qlike_ratio_vs_har']:.4f} | {r['qlike_regime']:.4f} | "
                         f"{r['mse_logrv']:.4f} | {r['coverage']:.3f} | {r['coverage_regime']:.3f} |")
        lines += ["", "| test | model vs | p | p_holm | passes |", "|---|---|---|---|---|"]
        for v, t in s["tests"].items():
            lines.append(f"| {v} | {t['model']} vs {t['vs']} | {t['p']:.4f} | {t['p_holm']:.4f} | {t['passes']} |")
        lines += ["", f"Range-engine candidates (pass + coverage in [0.75, 0.85]): {s['useful_range_engine'] or 'none'}", ""]
    return "\n".join(lines) + "\n"


def gvz_range(cfg: Config, raw, freqs) -> dict:
    """R-007: GVZ-implied normal range vs the cached R-003 GARCH range (G1) and R-006 daily GARCH (G2)."""
    from scipy.stats import norm

    from goldcast.config import ROOT

    qs = list(cfg.quantiles)
    z = norm.ppf(qs)
    out, fam = {}, {}
    for f in freqs:
        P = pd.read_csv(ROOT / "reports" / "vol" / f"{f}_predictions.csv", index_col=0, parse_dates=[0])
        t = pd.read_csv(ROOT / "reports" / "intervals" / f"{f}_predictions.csv", header=[0, 1], index_col=0, skiprows=[2])
        t.index = pd.to_datetime(t.index)
        idx = P.index.intersection(t.index)
        P, g = P.loc[idx], np.sort(t["garch"].loc[idx].values, axis=1)
        y = t["y"].iloc[:, 0].loc[idx].values if isinstance(t["y"], pd.DataFrame) else t["y"].loc[idx].values
        q = np.sqrt(P["gvz"].values)[:, None] * z[None, :]
        reg = idx >= pd.Timestamp(cfg.regime_start)
        pb_v, pb_g = M.pinball(y, q, qs), M.pinball(y, g, qs)
        ql_v, ql_g = qlike(P["rv"].values, P["gvz"].values), qlike(P["rv"].values, P["garch"].values)
        s = {"n": len(idx), "pinball_gvz": float(pb_v.mean()), "pinball_garch": float(pb_g.mean()),
             "pinball_ratio": float(pb_v.mean() / pb_g.mean()),
             "pinball_ratio_regime": float(pb_v[reg].mean() / pb_g[reg].mean()),
             "cov_gvz": M.coverage(y, q[:, 0], q[:, 2]), "cov_garch": M.coverage(y, g[:, 0], g[:, 2]),
             "cov_gvz_regime": M.coverage(y[reg], q[reg, 0], q[reg, 2]),
             "qlike_ratio": float(ql_v.mean() / ql_g.mean())}
        fam[(f, "G1")] = M.diebold_mariano(pb_g, pb_v)[1]
        fam[(f, "G2")] = M.diebold_mariano(ql_g, ql_v)[1]
        out[f] = s
    for (f, gname), p in M.holm(fam).items():
        s = out[f]
        ok = p < cfg.alpha and (gname == "G2" or 0.75 <= s["cov_gvz"] <= 0.85)
        s[gname] = {"p": fam[(f, gname)], "p_holm": p, "passes": bool(ok)}
    return out


def scaled_gvz_range(cfg: Config, freqs) -> dict:
    """R-008: GVZ variance times the expanding realized/implied ratio, vs the cached GARCH range."""
    from scipy.stats import norm

    from goldcast.config import ROOT

    qs = list(cfg.quantiles)
    z = norm.ppf(qs)
    out, fam = {}, {}
    for f in freqs:
        fs = cfg.freqs[f]
        F, y_rv, _ = frame(cfg, load_raw_cached(cfg), fs)
        imp = (F["gvz_level"] / 100) ** 2 / 252 * F["ndays"]
        ok = y_rv.notna() & imp.notna()
        ratio = (y_rv.where(ok).cumsum() / imp.where(ok).cumsum()).ffill().shift(1)  # rows j <= t-1
        t = pd.read_csv(ROOT / "reports" / "intervals" / f"{f}_predictions.csv", header=[0, 1], index_col=0, skiprows=[2])
        t.index = pd.to_datetime(t.index)
        idx = t.index.intersection(F.index[(F.index >= pd.Timestamp(cfg.test_start)) & y_rv.notna().values])
        g = np.sort(t["garch"].loc[idx].values, axis=1)
        yv = t["y"].loc[idx].values.ravel()
        k = ratio.loc[idx].values
        Fs = k * imp.loc[idx].values
        q = np.sqrt(Fs)[:, None] * z[None, :]
        reg = idx >= pd.Timestamp(cfg.regime_start)
        pb_s, pb_g = M.pinball(yv, q, qs), M.pinball(yv, g, qs)
        rv = y_rv.loc[idx].values
        fam[f] = M.diebold_mariano(pb_g, pb_s)[1]
        out[f] = {"n": len(idx), "pinball_ratio": float(pb_s.mean() / pb_g.mean()),
                  "pinball_ratio_regime": float(pb_s[reg].mean() / pb_g[reg].mean()),
                  "cov_scaled": M.coverage(yv, q[:, 0], q[:, 2]),
                  "cov_scaled_regime": M.coverage(yv[reg], q[reg, 0], q[reg, 2]),
                  "qlike_scaled_vs_gvz": float(qlike(rv, Fs).mean() / qlike(rv, imp.loc[idx].values).mean()),
                  "k_first": float(k[0]), "k_last": float(k[-1]), "p": fam[f]}
    for f, p in M.holm(fam).items():
        s = out[f]
        s["p_holm"] = p
        s["passes"] = bool(p < cfg.alpha and 0.75 <= s["cov_scaled"] <= 0.85)
    return out


def load_raw_cached(cfg):
    from goldcast.data.store import load_raw

    return load_raw(cfg)
