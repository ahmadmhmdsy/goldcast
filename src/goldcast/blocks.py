"""R-004: does one block of new series lower the R-002 LightGBM losses?

Arms: base = the R-002 series; Bk = base + block k; ALL = base + every block
(reported, not gated). Each arm runs lgbm (point) and lgbm_q (quantile) with the
R-002 parameters through the same walk-forward. Gates, Holm family and usefulness
checks follow the frozen pre-registration in AI_Research.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from goldcast.config import Config
from goldcast.evaluation import metrics as M
from goldcast.evaluation.walkforward import walk_forward
from goldcast.features.build import build_panel
from goldcast.models import REGISTRY
from goldcast.models.vol import GARCHQuantile

# Blocks left with no gold-specific series after the availability rule are reported
# but not gated (pre-registration: "B2: FX only" -> inconclusive).
UNGATED = {"B2": "inconclusive: only USDCNY and USDINR passed the availability rule"}
N_PERM = 20


def arms(cfg: Config) -> dict[str, tuple]:
    base = cfg.base_series
    out = {"base": base}
    for b in cfg.blocks:
        out[b] = base + tuple(s for s in cfg.series if s.block == b)
    out["ALL"] = cfg.series
    return out


def _limit(cfg, X, y, limit):
    if not limit:
        return X, y
    rows = np.flatnonzero((y.index >= pd.Timestamp(cfg.test_start)) & y.notna().values)
    last = y.index[rows[limit - 1]]
    return X.loc[:last], y.loc[:last]


def _run(cfg, fs, X, y, model):
    if model == "garch":
        fac = lambda: GARCHQuantile(cfg.quantiles)  # noqa: E731
    else:
        task = "point" if model == "lgbm" else "quantile"
        fac = lambda: REGISTRY[task][model](cfg, fs)  # noqa: E731
    return walk_forward(fac, X, y, cfg.test_start, fs.refit_every)


def _perm_importance(cfg, fs, X, y, cols, seed=0):
    """Static check: fit lgbm on pre-test rows, permute the block's columns jointly on the test rows."""
    test = (X.index >= pd.Timestamp(cfg.test_start)) & y.notna().values
    train = (X.index < pd.Timestamp(cfg.test_start)) & y.notna().values
    # purge the last training row: its target is the first test period's return
    train[np.flatnonzero(train)[-1]] = False
    m = REGISTRY["point"]["lgbm"](cfg, fs).fit(X[train], y[train])
    Xt, yt = X[test], y[test].values
    ref = np.mean((yt - m.predict(Xt)) ** 2)
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(N_PERM):
        Xp = Xt.copy()
        Xp[cols] = Xt[cols].values[rng.permutation(len(Xt))]
        deltas.append(np.mean((yt - m.predict(Xp)) ** 2) / ref - 1)
    return float(np.mean(deltas)), float(np.std(deltas))


def run_blocks(cfg: Config, raw: dict, freqs: list[str], limit: int | None = None, n_jobs: int = 8):
    t0, qs = time.time(), list(cfg.quantiles)
    lo, hi = qs.index(min(qs)), qs.index(max(qs))
    arm_specs = arms(cfg)
    panels = {}
    for f in freqs:
        for a, cat in arm_specs.items():
            X, y = build_panel(raw, cat, cfg.freqs[f], cfg.target)
            panels[(f, a)] = _limit(cfg, X, y, limit)
    jobs = [(f, a, m) for f in freqs for a in arm_specs for m in ("lgbm", "lgbm_q")]
    jobs += [(f, "base", "garch") for f in freqs]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_run)(cfg, cfg.freqs[f], *panels[(f, a)], m) for f, a, m in jobs)
    preds = dict(zip(jobs, res))
    print(f"  walk-forward done: {len(jobs)} runs, {time.time() - t0:.0f}s")

    scores, fam = {}, {}
    for f in freqs:
        fs = cfg.freqs[f]
        y = panels[(f, "base")][1]
        idx = preds[(f, "base", "lgbm")].index
        yt = y.loc[idx].values
        regime = idx >= pd.Timestamp(cfg.regime_start)
        rw = yt**2
        garch = M.pinball(yt, np.sort(preds[(f, "base", "garch")].loc[idx].values, axis=1), qs)
        se_b = (yt - preds[(f, "base", "lgbm")].loc[idx].values) ** 2
        pb_b = M.pinball(yt, preds[(f, "base", "lgbm_q")].loc[idx].values, qs)
        up = yt > 0
        s = {"n_test": int(len(idx)), "test_first": str(idx.min().date()), "test_last": str(idx.max().date()),
             "n_regime": int(regime.sum()),
             "rw": {"mse": float(rw.mean()), "mse_regime": float(rw[regime].mean())},
             "garch": {"pinball": float(garch.mean()), "pinball_regime": float(garch[regime].mean())},
             "hit_always_up": float(up.mean()), "arms": {}}
        for a in arm_specs:
            p = preds[(f, a, "lgbm")].loc[idx].values
            q = preds[(f, a, "lgbm_q")].loc[idx].values
            se, pb = (yt - p) ** 2, M.pinball(yt, q, qs)
            r = {"n_features": int(panels[(f, a)][0].shape[1]),
                 "mse": float(se.mean()), "mse_regime": float(se[regime].mean()),
                 "mse_ratio_vs_rw": float(se.mean() / rw.mean()),
                 "mse_ratio_vs_rw_regime": float(se[regime].mean() / rw[regime].mean()),
                 "pinball": float(pb.mean()), "pinball_regime": float(pb[regime].mean()),
                 "pinball_ratio_vs_garch": float(pb.mean() / garch.mean()),
                 "coverage": M.coverage(yt, q[:, lo], q[:, hi]),
                 "coverage_regime": M.coverage(yt[regime], q[regime, lo], q[regime, hi]),
                 "hit": float(((p > 0) == up).mean())}
            if a != "base":
                r["dm_point"], r["p_point"] = M.diebold_mariano(se_b, se)
                r["dm_quantile"], r["p_quantile"] = M.diebold_mariano(pb_b, pb)
                _, r["p_point_regime"] = M.diebold_mariano(se_b[regime], se[regime])
                _, r["p_quantile_regime"] = M.diebold_mariano(pb_b[regime], pb[regime])
                if a != "ALL":
                    X, yy = panels[(f, a)]
                    cols = [c for c in X.columns if c not in panels[(f, "base")][0].columns]
                    r["perm_mse_increase"], r["perm_sd"] = _perm_importance(cfg, fs, X, yy, cols)
                if a in cfg.blocks and a not in UNGATED:
                    fam[(f, a, "P")] = r["p_point"]
                    fam[(f, a, "Q")] = r["p_quantile"]
            s["arms"][a] = r
        scores[f] = s

    ph = M.holm(fam)
    alpha, gates = cfg.alpha, {}
    for b in cfg.blocks:
        if b in UNGATED:
            gates[b] = {"status": UNGATED[b]}
            continue
        g = {}
        for f in freqs:
            r = scores[f]["arms"][b]
            r["p_holm_point"], r["p_holm_quantile"] = ph[(f, b, "P")], ph[(f, b, "Q")]
            pk = bool(r["p_holm_point"] < alpha)
            qk = bool(r["p_holm_quantile"] < alpha and 0.75 <= r["coverage"] <= 0.85)
            g[f] = {"P_falsified": pk, "Q_falsified": qk,
                    "P_useful": bool(pk and r["mse_ratio_vs_rw"] < 1 and r["mse_ratio_vs_rw_regime"] < 1),
                    "Q_useful": bool(qk and r["pinball_ratio_vs_garch"] < 1)}
        g["any_gate"] = any(v[k] for v in g.values() for k in ("P_falsified", "Q_falsified"))
        g["useful"] = any(v[k] for v in g.values() if isinstance(v, dict) for k in ("P_useful", "Q_useful"))
        gates[b] = g
    gated = [g for g in gates.values() if "status" not in g]
    if len(UNGATED.keys() & set(cfg.blocks)) >= 2:
        verdict = "inconclusive"
    elif any(g["useful"] for g in gated):
        verdict = "supported"
    elif any(g["any_gate"] for g in gated):
        verdict = "partial"
    else:
        verdict = "refuted"
    print(f"  scored, {time.time() - t0:.0f}s")
    return {"verdict": verdict, "gates": gates, "holm_family_size": len(fam), "scores": scores}, preds


def to_markdown(out: dict) -> str:
    lines = [f"# R-004 blocks - verdict: **{out['verdict']}** (Holm family {out['holm_family_size']})", ""]
    for f, s in out["scores"].items():
        lines += [f"## {f} ({s['n_test']} origins, {s['test_first']}..{s['test_last']}; {s['n_regime']} in 2023+)",
                  f"RW mse {s['rw']['mse']:.6f}; GARCH pinball {s['garch']['pinball']:.6f}; "
                  f"always-up hit {s['hit_always_up']:.3f}", "",
                  "| arm | feats | mse/rw | mse/rw 2023+ | p pt | p_holm pt | pinball | pin/garch | cov | "
                  "p q | p_holm q | hit | perm dmse |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, r in s["arms"].items():
            def g(k, nd=4):
                v = r.get(k)
                return "-" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"
            lines.append(f"| {a} | {r['n_features']} | {g('mse_ratio_vs_rw')} | {g('mse_ratio_vs_rw_regime')} | "
                         f"{g('p_point')} | {g('p_holm_point')} | {g('pinball', 6)} | {g('pinball_ratio_vs_garch')} | "
                         f"{g('coverage', 3)} | {g('p_quantile')} | {g('p_holm_quantile')} | {g('hit', 3)} | "
                         f"{g('perm_mse_increase')} |")
        lines.append("")
    lines += ["## Gates", ""] + [f"- {b}: {g}" for b, g in out["gates"].items()]
    return "\n".join(lines) + "\n"
