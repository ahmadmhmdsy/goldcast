"""Command line: fetch | build | backtest | forecast."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from goldcast.config import ROOT, load_config
from goldcast.data.store import fetch_all, load_raw
from goldcast.evaluation.walkforward import apply_gates, run_freq, score_freq, to_markdown
from goldcast.features.build import build_panel
from goldcast.models import REGISTRY

REPORTS = ROOT / "reports"


def _freqs(cfg, which):
    return list(cfg.freqs) if which == "all" else [which]


def _panel(cfg, freq_name):
    raw = load_raw(cfg)
    return build_panel(raw, cfg.series, cfg.freqs[freq_name], cfg.target)


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def cmd_fetch(cfg, args):
    m = fetch_all(cfg, args.only)
    failed = [k for k, v in m.items() if "error" in v]
    if failed:
        print(f"{len(failed)} source(s) failed: {', '.join(failed)}")


def cmd_build(cfg, args):
    for f in _freqs(cfg, args.freq):
        X, y = _panel(cfg, f)
        print(f"{f}: {X.shape[0]} decision dates x {X.shape[1]} features, "
              f"{X.index.min().date()}..{X.index.max().date()}")
        cover = X.notna().mean().sort_values()
        print("  lowest feature coverage:", ", ".join(f"{k}={v:.2f}" for k, v in cover.head(5).items()))


def cmd_backtest(cfg, args):
    scores, t0 = {}, time.time()
    for f in _freqs(cfg, args.freq):
        X, y = _panel(cfg, f)
        preds = run_freq(cfg, cfg.freqs[f], X, y)
        scores[f] = score_freq(cfg, cfg.freqs[f], y, preds)
        out = REPORTS / f
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({f"{t}:{m}": (p if isinstance(p, pd.Series) else p.iloc[:, 1])
                      for t, d in preds.items() for m, p in d.items()}).assign(y=y).to_csv(
            out / "predictions.csv")
    gates = apply_gates(cfg, scores)
    for f, s in scores.items():
        (REPORTS / f / "scores.json").write_text(json.dumps(s, indent=2, default=_json_default))
        (REPORTS / f / "report.md").write_text(to_markdown(f, s), encoding="utf-8")
        print("\n" + to_markdown(f, s))
    (REPORTS / "gates.json").write_text(json.dumps(gates, indent=2, default=_json_default))
    print("## R-002 gates (Holm family: the frequencies run together)")
    for k, g in sorted(gates.items()):
        print(f"  {k}: falsified={g['falsified']}  {g['by'] or ''}  [{g['rule']}]")
    print(f"\nelapsed {time.time() - t0:.0f}s")


def cmd_forecast(cfg, args):
    """Fit on every row with a known target and forecast the next period from the latest row."""
    for f in _freqs(cfg, args.freq):
        X, y = _panel(cfg, f)
        known = y.notna()
        last = X.iloc[[-1]]
        print(f"{f}: forecast for the period after {last.index[0].date()}")
        for task, models in REGISTRY.items():
            for name, factory in models.items():
                p = np.asarray(factory(cfg, cfg.freqs[f]).fit(X[known], y[known]).predict(last))
                print(f"  {task:<9} {name:<9} {np.round(p.ravel(), 5).tolist()}")


def cmd_chronos(cfg, args):
    """Chronos-2 arm: its own Holm family against rw, qbase and always-up (R-002 dated note)."""
    from goldcast.evaluation import metrics as M
    from goldcast.evaluation.walkforward import walk_forward
    from goldcast.models.foundation import chronos_forecasts, load_pipeline

    pipe, t0, scores = load_pipeline(args.threads), time.time(), {}
    for f in _freqs(cfg, args.freq):
        fs = cfg.freqs[f]
        X, y = _panel(cfg, f)
        q = chronos_forecasts(pipe, y, cfg.test_start, cfg.quantiles, limit=args.limit)
        X, y = X.loc[: q.index.max()], y.loc[: q.index.max()]
        mid = q.iloc[:, list(cfg.quantiles).index(0.5)]
        base = {t: walk_forward(lambda t=t: REGISTRY[t][m](cfg, fs), X, y, cfg.test_start, fs.refit_every)
                for t, m in (("point", "rw"), ("quantile", "qbase"), ("direction", "always_up"))}
        preds = {"point": {"rw": base["point"], "chronos": mid},
                 "quantile": {"qbase": base["quantile"], "chronos": q},
                 "direction": {"always_up": base["direction"], "chronos": (mid > 0).astype(float)}}
        scores[f] = score_freq(cfg, fs, y, preds)
        print(f"  {f}: {len(q)} forecasts, {time.time() - t0:.0f}s")
    fam = {}
    for f, s in scores.items():
        fam[(f, "point")] = s["point"]["chronos"]["p"]
        fam[(f, "direction")] = s["direction"]["chronos"]["p"]
        fam[(f, "quantile")] = s["quantile"]["chronos"]["p_vs_qbase"]
    for (f, t), p in M.holm(fam).items():
        scores[f][t]["chronos"]["p_holm"] = p
    a, gates = cfg.alpha, {}
    for f, s in scores.items():
        pt, dr, qt = s["point"]["chronos"], s["direction"]["chronos"], s["quantile"]
        gates[f] = {
            "C1": bool(pt["p_holm"] < a and pt["mse_ratio_regime"] < 1),
            "C2": bool(qt["chronos"]["pinball"] < qt["qbase"]["pinball"] and 0.75 <= qt["chronos"]["coverage"] <= 0.85),
            "C3": bool(dr["p_holm"] < a and dr["hit"] > dr["hit_always_up"]),
        }
    out = REPORTS / ("chronos_smoke" if args.limit else "chronos")
    out.mkdir(parents=True, exist_ok=True)
    for f, s in scores.items():
        (out / f"{f}.md").write_text(to_markdown(f, s), encoding="utf-8")
        print("\n" + to_markdown(f, s))
    (out / "gates.json").write_text(json.dumps({"passed": gates, "scores": scores}, indent=2, default=_json_default))
    print("Chronos gates passed (True = hypothesis of no skill rejected):", gates)
    print(f"elapsed {time.time() - t0:.0f}s")


def cmd_intervals(cfg, args):
    """R-003: 80% intervals against GARCH(1,1)-t, one Holm family of V1-V3 x horizons."""
    from goldcast.evaluation import metrics as M
    from goldcast.evaluation.walkforward import walk_forward
    from goldcast.models.foundation import chronos_forecasts, load_pipeline
    from goldcast.models.vol import EWMAQuantile, GARCHQuantile

    qs, t0, scores, fam = list(cfg.quantiles), time.time(), {}, {}
    lo, hi = qs.index(min(qs)), qs.index(max(qs))
    pipe = None if args.no_chronos else load_pipeline(args.threads)
    out = REPORTS / ("intervals_smoke" if args.limit else "intervals")
    out.mkdir(parents=True, exist_ok=True)
    for f in _freqs(cfg, args.freq):
        fs = cfg.freqs[f]
        X, y = _panel(cfg, f)
        if args.limit:
            last = y.index[np.flatnonzero((y.index >= pd.Timestamp(cfg.test_start)) & y.notna().values)[args.limit - 1]]
            X, y = X.loc[:last], y.loc[:last]
        factories = {"qbase": lambda: REGISTRY["quantile"]["qbase"](cfg, fs),
                     "ewma": lambda: EWMAQuantile(cfg.quantiles),
                     "garch": lambda: GARCHQuantile(cfg.quantiles),
                     "lgbm_q": lambda: REGISTRY["quantile"]["lgbm_q"](cfg, fs)}
        preds = {}
        for name, fac in factories.items():
            p = walk_forward(fac, X, y, cfg.test_start, fs.refit_every)
            preds[name] = pd.DataFrame(np.sort(p.values, axis=1), index=p.index, columns=qs)
            print(f"  {f:<7} {name:<7} {len(p)} forecasts, {time.time() - t0:.0f}s")
        if pipe is not None:
            preds["chronos"] = chronos_forecasts(pipe, y, cfg.test_start, cfg.quantiles)
            print(f"  {f:<7} chronos {len(preds['chronos'])} forecasts, {time.time() - t0:.0f}s")
        idx = preds["garch"].index
        yt = y.loc[idx].values
        regime = idx >= pd.Timestamp(cfg.regime_start)
        ref = M.pinball(yt, preds["garch"].values, qs)
        s = {}
        for name, p in preds.items():
            v = p.loc[idx].values
            loss = M.pinball(yt, v, qs)
            row = {"pinball": float(loss.mean()), "pinball_regime": float(loss[regime].mean()),
                   "coverage": M.coverage(yt, v[:, lo], v[:, hi]),
                   "coverage_regime": M.coverage(yt[regime], v[regime, lo], v[regime, hi])}
            if name != "garch":  # one-sided: is `name` better than GARCH?
                row["dm_vs_garch"], row["p_vs_garch"] = M.diebold_mariano(ref, loss)
            s[name] = row
        # V1 asks whether GARCH beats qbase: the reverse direction of the qbase row's test.
        _, s["garch"]["p_vs_qbase"] = M.diebold_mariano(M.pinball(yt, preds["qbase"].loc[idx].values, qs), ref)
        fam[(f, "V1")] = s["garch"]["p_vs_qbase"]
        fam[(f, "V3")] = s["lgbm_q"]["p_vs_garch"]
        if "chronos" in s:
            fam[(f, "V2")] = s["chronos"]["p_vs_garch"]
        scores[f] = {"n_test": int(len(idx)), "n_regime": int(regime.sum()), "models": s}
        pd.concat({k: v.loc[idx] for k, v in preds.items()}, axis=1).assign(y=yt).to_csv(out / f"{f}_predictions.csv")
    ph = M.holm(fam)
    a, gates = cfg.alpha, {}
    for f, sc in scores.items():
        s, g = sc["models"], {}
        g["V1_holds"] = bool(s["garch"]["pinball"] < s["qbase"]["pinball"] and ph[(f, "V1")] < a)
        for v, m in (("V2", "chronos"), ("V3", "lgbm_q")):
            if (f, v) in ph:
                beat = (s[m]["pinball"] < s["garch"]["pinball"] and ph[(f, v)] < a
                        and 0.75 <= s[m]["coverage"] <= 0.85)
                g[f"{v}_falsified"] = bool(beat)
        sc["p_holm"] = {v: ph[(ff, v)] for (ff, v) in ph if ff == f}
        gates[f] = g
        print(f"\n## {f} ({sc['n_test']} origins; {sc['n_regime']} in 2023+)")
        print("| model | pinball | pinball 2023+ | coverage | coverage 2023+ | DM p vs garch |")
        print("|---|---|---|---|---|---|")
        for m, r in s.items():
            pv = r.get("p_vs_garch", float("nan"))
            print(f"| {m} | {r['pinball']:.6f} | {r['pinball_regime']:.6f} | {r['coverage']:.3f} "
                  f"| {r['coverage_regime']:.3f} | {pv:.4f} |")
        print("Holm p:", {k: round(v, 4) for k, v in sc["p_holm"].items()})
    (out / "gates.json").write_text(json.dumps({"gates": gates, "scores": scores}, indent=2, default=_json_default))
    print("\nR-003 gates:", gates)
    print(f"elapsed {time.time() - t0:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="goldcast")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--only", nargs="*")
    for name in ("build", "backtest", "forecast"):
        p = sub.add_parser(name)
        p.add_argument("--freq", default="all", choices=["weekly", "monthly", "all"])
    p = sub.add_parser("chronos")
    p.add_argument("--freq", default="all", choices=["weekly", "monthly", "all"])
    p.add_argument("--limit", type=int, default=None, help="smoke run: first N test rows only")
    p.add_argument("--threads", type=int, default=10)
    p = sub.add_parser("intervals")
    p.add_argument("--freq", default="all", choices=["weekly", "monthly", "all"])
    p.add_argument("--limit", type=int, default=None, help="smoke run: first N test rows only")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--no-chronos", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    {"fetch": cmd_fetch, "build": cmd_build, "backtest": cmd_backtest,
     "forecast": cmd_forecast, "chronos": cmd_chronos, "intervals": cmd_intervals}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
