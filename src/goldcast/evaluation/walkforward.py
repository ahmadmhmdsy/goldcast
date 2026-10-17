"""Expanding-window walk-forward with purging, plus the report that scores it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goldcast.config import Config, FreqSpec
from goldcast.evaluation import metrics as M
from goldcast.models import BASELINES, REGISTRY


def walk_forward(model_factory, X: pd.DataFrame, y: pd.Series, test_start: str,
                 refit_every: int, h: int = 1) -> pd.DataFrame | pd.Series:
    """Forecast every row from `test_start` on, fitting only on rows whose target is known.

    Row j's target is known at row j + h, so a model used at origin i trains on
    rows j <= i - h. Cheap models refit every row; others every `refit_every` rows.
    """
    rows = np.flatnonzero((X.index >= pd.Timestamp(test_start)) & y.notna().values)
    if rows.size == 0:
        raise ValueError("no test rows")
    probe = model_factory()
    step = 1 if getattr(probe, "cheap", False) else refit_every
    preds = {}
    for k in range(0, rows.size, step):
        block = rows[k:k + step]
        i = block[0]
        tr = slice(0, i - h + 1)
        ytr = y.iloc[tr]
        ok = ytr.notna().values
        model = model_factory().fit(X.iloc[tr][ok], ytr[ok])
        out = model.predict(X.iloc[block])
        for j, r in enumerate(block):
            preds[X.index[r]] = out[j]
    idx = pd.DatetimeIndex(list(preds), name="date")
    vals = np.array(list(preds.values()))
    return pd.DataFrame(vals, index=idx) if vals.ndim == 2 else pd.Series(vals, index=idx)


def run_freq(cfg: Config, freq: FreqSpec, X: pd.DataFrame, y: pd.Series) -> dict:
    """Run every registered model for one frequency; return raw predictions by task."""
    preds = {}
    for task, models in REGISTRY.items():
        preds[task] = {}
        for name, factory in models.items():
            preds[task][name] = walk_forward(lambda: factory(cfg, freq), X, y,
                                             cfg.test_start, freq.refit_every)
            print(f"  {freq.name:<7} {task:<9} {name:<9} {len(preds[task][name])} forecasts")
    return preds


def score_freq(cfg: Config, freq: FreqSpec, y: pd.Series, preds: dict) -> dict:
    """Unadjusted statistics for one frequency. Holm is applied across frequencies later."""
    yt = y.loc[preds["point"]["rw"].index].values
    idx = preds["point"]["rw"].index
    regime = idx >= pd.Timestamp(cfg.regime_start)
    out = {"n_test": int(len(yt)), "test_first": str(idx.min().date()),
           "test_last": str(idx.max().date()), "n_regime": int(regime.sum()),
           "point": {}, "direction": {}, "quantile": {}}

    base_loss = yt**2  # random walk forecasts zero
    for name, p in preds["point"].items():
        e2 = (yt - p.values) ** 2
        dm, pv = M.diebold_mariano(base_loss, e2) if name != "rw" else (np.nan, np.nan)
        _, pv_reg = M.diebold_mariano(base_loss[regime], e2[regime]) if name != "rw" else (np.nan, np.nan)
        out["point"][name] = {
            "mse": float(e2.mean()), "mae": float(np.abs(yt - p.values).mean()),
            "mse_ratio_vs_rw": float(e2.mean() / base_loss.mean()),
            "dm": dm, "p": pv,
            "mse_regime": float(e2[regime].mean()),
            "mse_ratio_regime": float(e2[regime].mean() / base_loss[regime].mean()),
            "p_regime": pv_reg,
        }

    up = yt > 0
    hit_up = float(up.mean())
    for name, p in preds["direction"].items():
        call = p.values > 0.5
        hit = float((call == up).mean())
        pt, pv = M.pesaran_timmermann(up, call) if name != "always_up" else (np.nan, np.nan)
        strat = M.long_flat(yt, call, cfg.cost_bp, freq.periods_per_year)
        out["direction"][name] = {
            "hit": hit, "hit_always_up": hit_up, "pt": pt, "p": pv,
            "hit_regime": float((call[regime] == up[regime]).mean()),
            "strategy": strat,
        }

    qs = list(cfg.quantiles)
    lo_i, hi_i = qs.index(min(qs)), qs.index(max(qs))
    qbase_loss = M.pinball(yt, preds["quantile"]["qbase"].values, qs)
    for name, p in preds["quantile"].items():
        loss = M.pinball(yt, p.values, qs)
        dm, pv = M.diebold_mariano(qbase_loss, loss) if name != "qbase" else (np.nan, np.nan)
        out["quantile"][name] = {
            "pinball": float(loss.mean()),
            "pinball_regime": float(loss[regime].mean()),
            "coverage": M.coverage(yt, p.values[:, lo_i], p.values[:, hi_i]),
            "coverage_regime": M.coverage(yt[regime], p.values[regime, lo_i], p.values[regime, hi_i]),
            "dm_vs_qbase": dm, "p_vs_qbase": pv,
        }
    return out


def apply_gates(cfg: Config, scores: dict[str, dict]) -> dict:
    """Holm across models x horizons, then the R-002 falsification checks."""
    a = cfg.alpha
    p_point = {(f, m): s["point"][m]["p"] for f, s in scores.items() for m in s["point"]}
    p_dir = {(f, m): s["direction"][m]["p"] for f, s in scores.items() for m in s["direction"]}
    for fam, task in ((M.holm(p_point), "point"), (M.holm(p_dir), "direction")):
        for (f, m), p in fam.items():
            scores[f][task][m]["p_holm"] = p

    h1_hits = [f"{f}:{m}" for f, s in scores.items() for m, r in s["point"].items()
               if m != "rw" and np.isfinite(r["p_holm"]) and r["p_holm"] < a
               and r["mse_ratio_regime"] < 1]
    h3_hits = [f"{f}:{m}" for f, s in scores.items() for m, r in s["direction"].items()
               if m != "always_up" and np.isfinite(r["p_holm"]) and r["p_holm"] < a
               and r["hit"] > r["hit_always_up"]]
    gates = {
        "H1": {"falsified": bool(h1_hits), "by": h1_hits,
               "rule": "point model beats rw: DM p_holm < alpha AND lower MSE on 2023+"},
        "H3": {"falsified": bool(h3_hits), "by": h3_hits,
               "rule": "direction model hit > always-up AND PT p_holm < alpha"},
    }
    if "weekly" in scores:
        q = scores["weekly"]["quantile"]
        lo, hi = 0.75, 0.85
        fails = []
        if not q["lgbm_q"]["pinball"] < q["qbase"]["pinball"]:
            fails.append("pinball not below qbase")
        if not lo <= q["lgbm_q"]["coverage"] <= hi:
            fails.append(f"coverage {q['lgbm_q']['coverage']:.3f} outside [{lo}, {hi}]")
        gates["H2"] = {"falsified": bool(fails), "by": fails,
                       "rule": "weekly lgbm_q pinball < qbase AND 80% coverage in [0.75, 0.85]"}
    else:
        gates["H2"] = {"falsified": None, "by": [], "rule": "needs the weekly run"}
    return gates


def _f(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def to_markdown(freq: str, s: dict) -> str:
    L = [f"# Backtest — {freq}", "",
         f"Test window {s['test_first']} .. {s['test_last']} ({s['n_test']} periods; "
         f"{s['n_regime']} in the 2023+ sub-window).", "",
         "## Point forecasts (log return)", "",
         "| model | MSE/rw | DM | p | p Holm | MSE/rw 2023+ | MAE |",
         "|---|---|---|---|---|---|---|"]
    for m, r in s["point"].items():
        L.append(f"| {m} | {_f(r['mse_ratio_vs_rw'])} | {_f(r['dm'], 2)} | {_f(r['p'])} | "
                 f"{_f(r.get('p_holm'))} | {_f(r['mse_ratio_regime'])} | {_f(r['mae'], 5)} |")
    L += ["", "## Direction", "",
          "| model | hit | always-up hit | PT p | p Holm | hit 2023+ | strat Sharpe | B&H Sharpe | time long |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, r in s["direction"].items():
        st = r["strategy"]
        L.append(f"| {m} | {_f(r['hit'], 3)} | {_f(r['hit_always_up'], 3)} | {_f(r['p'])} | "
                 f"{_f(r.get('p_holm'))} | {_f(r['hit_regime'], 3)} | {_f(st['strategy']['sharpe'], 2)} | "
                 f"{_f(st['buy_hold']['sharpe'], 2)} | {_f(st['time_long'], 2)} |")
    L += ["", "## 80% interval (q = 0.1 / 0.5 / 0.9)", "",
          "| model | mean pinball | pinball 2023+ | coverage | coverage 2023+ | DM p vs qbase |",
          "|---|---|---|---|---|---|"]
    for m, r in s["quantile"].items():
        L.append(f"| {m} | {_f(r['pinball'], 6)} | {_f(r['pinball_regime'], 6)} | "
                 f"{_f(r['coverage'], 3)} | {_f(r['coverage_regime'], 3)} | {_f(r['p_vs_qbase'])} |")
    return "\n".join(L) + "\n"
