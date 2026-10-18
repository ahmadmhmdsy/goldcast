"""R-005: foundation-model arms (TabPFN-2, TabFM, Chronos-2 with past covariates) on the R-002 panel.

Each arm's forecasts are cached per frequency in the output directory, so arms can be
run one at a time and scored together. References (lgbm, lgbm_q, GARCH) are rerun with
the R-002/R-003 code; the random walk forecasts zero.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from goldcast.blocks import _limit, _run
from goldcast.config import Config
from goldcast.evaluation import metrics as M
from goldcast.evaluation.walkforward import walk_forward
from goldcast.features.build import build_panel

ARMS = ("tabpfn", "tabfm", "chronos2x")
POINT_ARMS, QUANT_ARMS = ("tabpfn", "tabfm", "chronos2x"), ("tabpfn", "chronos2x")


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class TabPFNArm:
    """TabPFN-2 regressor; predict returns [mean, q...] per row."""

    def __init__(self, quantiles):
        self.qs = list(quantiles)

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        self.m = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device=device(), random_state=0, ignore_pretraining_limits=True)
        self.m.fit(X.values, y.values)
        return self

    def predict(self, X):
        mean = self.m.predict(X.values)
        q = np.column_stack(self.m.predict(X.values, output_type="quantiles", quantiles=self.qs))
        return np.column_stack([mean, np.sort(q, axis=1)])


_TABFM = {}


class TabFMArm:
    """TabFM v1.0.0 regressor, package defaults; point only. Weights loaded once per process."""

    def fit(self, X, y):
        from tabfm import TabFMRegressor
        from tabfm import tabfm_v1_0_0_pytorch as T

        if "m" not in _TABFM:
            _TABFM["m"] = T.load(model_type="regression").to(device())
        # TabFM rejects NaN; fill with training-row medians (all-NaN columns -> 0)
        self.med = X.median().fillna(0.0)
        self.m = TabFMRegressor(model=_TABFM["m"]).fit(X.fillna(self.med).values, y.values)
        return self

    def predict(self, X):
        return np.asarray(self.m.predict(X.fillna(self.med).values), float)


def chronos_covariates(X: pd.DataFrame) -> pd.DataFrame:
    """One-period change of each non-gold base series, shifted so position j covers target j's period."""
    cols = [c for c in X.columns if not c.startswith("gold_") and c.endswith(("logret1", "diff1"))]
    return X[cols].shift(-1)


def chronos2x_forecasts(pipe, X, y, test_start, quantiles, limit=None, batch_size=32):
    """Context at origin i: targets y[:i] and covariates cov[:i]; every value is known at i."""
    cov = chronos_covariates(X)
    rows = np.flatnonzero((y.index >= pd.Timestamp(test_start)) & y.notna().values)
    if limit:
        rows = rows[:limit]
    start = int(np.flatnonzero(cov.notna().all(axis=1).values & y.notna().values)[0])
    tasks = []
    for i in rows:
        c = cov.iloc[start:i]
        tasks.append({"target": y.iloc[start:i].to_numpy(np.float32),
                      "past_covariates": {k: c[k].to_numpy(np.float32) for k in c.columns}})
    out = []
    for k in range(0, len(tasks), batch_size):
        q, _ = pipe.predict_quantiles(tasks[k:k + batch_size], prediction_length=1,
                                      quantile_levels=list(quantiles))
        out += [t[0, 0, :].float().numpy() for t in q]
    q = np.sort(np.array(out), axis=1)
    return pd.DataFrame(np.column_stack([q[:, list(quantiles).index(0.5)], q]),
                        index=y.index[rows], columns=["point", *quantiles])


def run_arm(cfg: Config, raw, f: str, arm: str, limit=None, threads=None) -> pd.DataFrame:
    fs = cfg.freqs[f]
    X, y = _limit(cfg, *build_panel(raw, cfg.base_series, fs, cfg.target), limit)
    qs = list(cfg.quantiles)
    if arm == "chronos2x":
        from goldcast.models.foundation import load_pipeline
        return chronos2x_forecasts(load_pipeline(threads), X, y, cfg.test_start, qs, limit)
    if arm == "tabpfn":
        p = walk_forward(lambda: TabPFNArm(qs), X, y, cfg.test_start, fs.refit_every)
        p.columns = ["point", *qs]
        return p
    p = walk_forward(TabFMArm, X, y, cfg.test_start, fs.refit_every)
    return pd.DataFrame({"point": p})


def score(cfg: Config, raw, freqs, cached: dict, limit=None) -> dict:
    """cached[(f, arm)] = DataFrame with 'point' and, for quantile arms, the quantile columns."""
    qs = list(cfg.quantiles)
    lo, hi = qs.index(min(qs)), qs.index(max(qs))
    scores, fam = {}, {}
    for f in freqs:
        fs = cfg.freqs[f]
        X, y = _limit(cfg, *build_panel(raw, cfg.base_series, fs, cfg.target), limit)
        ref = {m: _run(cfg, fs, X, y, m) for m in ("lgbm", "lgbm_q", "garch")}
        idx = ref["lgbm"].index
        yt = y.loc[idx].values
        reg = idx >= pd.Timestamp(cfg.regime_start)
        up = yt > 0
        loss = {"rw": yt**2, "lgbm": (yt - ref["lgbm"].loc[idx].values) ** 2,
                "lgbm_q": M.pinball(yt, ref["lgbm_q"].loc[idx].values, qs),
                "garch": M.pinball(yt, np.sort(ref["garch"].loc[idx].values, axis=1), qs)}
        s = {"n_test": len(idx), "test_first": str(idx.min().date()), "test_last": str(idx.max().date()),
             "n_regime": int(reg.sum()), "hit_always_up": float(up.mean()),
             "refs": {k: {"loss": float(v.mean()), "loss_regime": float(v[reg].mean())} for k, v in loss.items()},
             "arms": {}}
        for arm in ARMS:
            if (f, arm) not in cached:
                continue
            P = cached[(f, arm)].loc[idx]
            se = (yt - P["point"].values) ** 2
            r = {"mse": float(se.mean()), "mse_ratio_vs_rw": float(se.mean() / loss["rw"].mean()),
                 "mse_ratio_vs_rw_regime": float(se[reg].mean() / loss["rw"][reg].mean()),
                 "mse_ratio_vs_lgbm": float(se.mean() / loss["lgbm"].mean()),
                 "hit": float(((P["point"].values > 0) == up).mean())}
            for rf in ("lgbm", "rw"):
                _, r[f"p_point_vs_{rf}"] = M.diebold_mariano(loss[rf], se)
                fam[(f, arm, "P", rf)] = r[f"p_point_vs_{rf}"]
            if arm in QUANT_ARMS:
                q = P[qs].values
                pb = M.pinball(yt, q, qs)
                r.update({"pinball": float(pb.mean()), "pinball_regime": float(pb[reg].mean()),
                          "pinball_ratio_vs_garch": float(pb.mean() / loss["garch"].mean()),
                          "pinball_ratio_vs_lgbm_q": float(pb.mean() / loss["lgbm_q"].mean()),
                          "coverage": M.coverage(yt, q[:, lo], q[:, hi])})
                for rf in ("lgbm_q", "garch"):
                    _, r[f"p_quantile_vs_{rf}"] = M.diebold_mariano(loss[rf], pb)
                    fam[(f, arm, "Q", rf)] = r[f"p_quantile_vs_{rf}"]
            s["arms"][arm] = r
        scores[f] = s

    ph = M.holm(fam)
    gates = {}
    for (f, arm, t, rf), p in ph.items():
        r = scores[f]["arms"][arm]
        key = f"{'point' if t == 'P' else 'quantile'}_vs_{rf}"
        r[f"p_holm_{key}"] = p
        ok = p < cfg.alpha and (t == "P" or 0.75 <= r["coverage"] <= 0.85)
        gates[f"{f}:{arm}:{key}"] = bool(ok)
    useful = []
    for f, s in scores.items():
        for arm, r in s["arms"].items():
            if gates.get(f"{f}:{arm}:point_vs_rw") and r["mse_ratio_vs_rw_regime"] < 1:
                useful.append(f"{f}:{arm}:point")
            if gates.get(f"{f}:{arm}:quantile_vs_garch"):
                useful.append(f"{f}:{arm}:quantile")
    verdict = "supported" if useful else ("partial" if any(gates.values()) else "refuted")
    return {"verdict": verdict, "useful": useful, "gates": gates, "holm_family_size": len(fam),
            "scores": scores}


def to_markdown(out: dict) -> str:
    lines = [f"# R-005 foundation arms - verdict: **{out['verdict']}** (Holm family {out['holm_family_size']})", ""]
    for f, s in out["scores"].items():
        rf = s["refs"]
        lines += [f"## {f} ({s['n_test']} origins, {s['test_first']}..{s['test_last']}; {s['n_regime']} in 2023+)",
                  f"RW mse {rf['rw']['loss']:.6f}; lgbm mse {rf['lgbm']['loss']:.6f}; "
                  f"lgbm_q pinball {rf['lgbm_q']['loss']:.6f}; GARCH pinball {rf['garch']['loss']:.6f}; "
                  f"always-up hit {s['hit_always_up']:.3f}", "",
                  "| arm | mse/rw | mse/rw 2023+ | mse/lgbm | p_holm vs lgbm | p_holm vs rw | pinball | "
                  "pin/garch | pin/lgbm_q | cov | p_holm vs lgbm_q | p_holm vs garch | hit |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, r in s["arms"].items():
            def g(k, nd=4):
                v = r.get(k)
                return "-" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"
            lines.append(f"| {a} | {g('mse_ratio_vs_rw')} | {g('mse_ratio_vs_rw_regime')} | {g('mse_ratio_vs_lgbm')} | "
                         f"{g('p_holm_point_vs_lgbm')} | {g('p_holm_point_vs_rw')} | {g('pinball', 6)} | "
                         f"{g('pinball_ratio_vs_garch')} | {g('pinball_ratio_vs_lgbm_q')} | {g('coverage', 3)} | "
                         f"{g('p_holm_quantile_vs_lgbm_q')} | {g('p_holm_quantile_vs_garch')} | {g('hit', 3)} |")
        lines.append("")
    lines += ["## Gates", ""] + [f"- {k}: {v}" for k, v in out["gates"].items()]
    lines += ["", f"Useful: {out['useful'] or 'none'}"]
    return "\n".join(lines) + "\n"


def timed(fn, *a, **k):
    t = time.time()
    return fn(*a, **k), time.time() - t
