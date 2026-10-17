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
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    {"fetch": cmd_fetch, "build": cmd_build, "backtest": cmd_backtest,
     "forecast": cmd_forecast, "chronos": cmd_chronos}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
