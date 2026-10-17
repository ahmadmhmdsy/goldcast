# goldcast

Weekly and monthly gold forecasts from free cross-asset and macro data, scored
honestly against naive baselines. Pre-registration and verdict:
`AI_Research/research/experiments/R-002-gold-forecasting.md`.

## First result (2026-10-17, test window 2015-01 .. 2026-10)

Nothing beat the naive baselines. That matches the literature (Hassani et al. 2015).

| gate | outcome |
|---|---|
| H1: no point model beats the random walk | **held**. Best weekly MSE/rw = 0.992 (drift); best monthly = 0.967 (drift). Every Holm-adjusted DM p = 1.0. Ridge and LightGBM were worse than the random walk. |
| H2: LightGBM quantile beats the rolling-quantile baseline (weekly) | **failed**. Pinball 0.005421 vs 0.005405, coverage 0.785. It is slightly better on 2023+ only, and that window was not gated. |
| H3: no direction model beats always-up | **held**. Always-up hit rate is 0.553 weekly and 0.557 monthly. The best model was the monthly LightGBM classifier at 0.564, with PT p (Holm) = 0.55. |

Full tables: `reports/<freq>/report.md`; gate verdicts: `reports/gates.json`.

### Chronos-2 zero-shot arm (2026-10-17, CPU, 83 s)

`python -m goldcast chronos --freq all` runs `amazon/chronos-2` on gold's own return history. It is scored against rw, qbase and always-up, as a separate Holm family. Weights are about 460 MB, and CPU torch must be installed in the venv.

| gate | weekly | monthly |
|---|---|---|
| C1 point (MSE/rw, Holm p) | 0.998, 1.0 — fail | 0.980, 1.0 — fail |
| C2 interval (pinball vs qbase, coverage) | 0.005345 vs 0.005405, 0.807 — pass | 0.010934 vs 0.010930, 0.757 — fail |
| C3 direction | median always > 0, so it acts as always-up — fail | 0.550 vs 0.557 — fail |

The weekly interval gain is small, at DM p = 0.068 unadjusted, and comes from a single run. Tables are in `reports/chronos/`.

## Layout

    configs/default.toml        series catalog, publication lags, windows, test window
    src/goldcast/data/          fetchers (Yahoo, FRED csv, GPR xls) and parquet store + manifest
    src/goldcast/features/      point-in-time panel: values moved to availability date, as-of join
    src/goldcast/models/        baselines, ridge/logit, LightGBM point/quantile/classifier, registry;
                                foundation.py = Chronos-2 zero-shot arm
    src/goldcast/evaluation/    walk-forward (expanding, purged), DM, PT, pinball, Holm, long/flat
    tests/                      truncation leakage test, metrics, walk-forward purging (offline)

## Use

    uv venv .venv && uv pip install -e ".[dev]"
    python -m goldcast fetch            # data/raw/*.parquet + manifest.json
    python -m goldcast build            # panel shapes and feature coverage
    python -m goldcast backtest         # both frequencies, about 6 min on CPU
    python -m goldcast forecast         # next-period forecasts from every model
    python -m pytest

## Rules the code enforces

- Each value enters the panel on the date it became public, not the date it describes.
  The lags are: market closes 0 business days, FRED daily 1, GPR 7.
  The test `tests/test_no_leakage.py` builds the panel from truncated data and from poisoned future data. Both must match the full build.
- The target is the log return from one decision close to the next. It is never the price level.
- Training rows are purged, so no training target ends after the forecast origin.
- Imputers and scalers are fit on the training rows only.
- Every model uses the same splits. The baselines are refit every period, so they are never stale.

## Adding things

- **A series:** add a `[[series]]` row in `configs/default.toml` with its source, lag and kinds.
- **A source:** add a fetcher to `FETCHERS`.
- **A model:** add a class with `fit(X, y)` / `predict(X)` and register it in `goldcast/models/__init__.py`.
- **A changed gate:** gates are frozen in R-002. A change goes in a dated note there, not here.
