# goldcast 🪙

**English** | [العربية](README.ar.md)

**Honest gold-price forecasting.** Eleven pre-registered studies, about 20 models (from a random walk to
Chronos-2, TabPFN-2 and TabFM), ten years of walk-forward backtests, and a dashboard that tells you
what gold might do next week, next month or in two years, and how much to trust it.

![The Forecast tab: 1-month forecast, 80% range and model comparison](docs/forecast_tab.jpg)

## Why this project is different

Most gold forecasting repos show a fitted line that hugs the price chart. This one does the opposite:

- **Every study is pre-registered.** Hypotheses, gates and the test window are frozen before any run,
  and a failed gate stays failed.
- **No look-ahead.** Each value enters the panel on the date it became public, not the date it
  describes. A test rebuilds the panel from truncated and from poisoned future data, and both must
  match the full build.
- **Everything is scored against naive baselines.** Point forecasts face the random walk
  (Diebold-Mariano), ranges face historical quantiles and GARCH (pinball loss), and direction faces
  always-up (Pesaran-Timmermann). Holm correction is applied across each model family.
- **Negative results are reported.** As the literature predicts, almost nothing beats the baselines,
  and the README says so.

## The dashboard

```bash
pip install -e ".[dashboard]"
streamlit run src/goldcast/dashboard/app.py      # or double-click run_dashboard.bat on Windows
```

- **🔮 Forecast.** Pick a horizon: 1 week, 1 month, 3 months, 6 months, 1 year or 2 years.
  - The model with the lowest backtest error at that horizon is selected by default (⭐). You can
    switch to any other model.
  - You see the forecast price, an 80% range, the chance gold ends higher, and the model's
    backtest record.
  - A price cone, a side-by-side table of every model, and a track-record chart of past forecasts
    against what actually happened.
- **Leaderboard, Forecast vs actual, Error over time.** Every model from every study, scored on any
  window (full test, 2023+, or custom), with significance after Holm correction.
- **Input data, Next period.** The point-in-time feature panel, and live next-period forecasts from the
  cheap models, plus the GVZ implied move.
- **Update prices.** Re-downloads all series in one click.

## What we found (test window 2015-01 to 2026-10)

| Study | Question | Verdict |
|---|---|---|
| R-002 | Do 13 free cross-asset and macro series beat a random walk (weekly, monthly)? | **No.** Best MSE/rw is 0.99 weekly and 0.97 monthly (drift). Ridge and LightGBM are worse than the random walk. |
| R-003 | Is the interval edge more than volatility tracking? | **Partial.** GARCH and EWMA give the best 80% ranges; no gain survives Holm. |
| R-004 | Do flows/positioning, China-India FX, uncertainty or Google Trends help? | **Refuted.** No block lowers the loss. |
| R-005 | Do foundation models (TabPFN-2, TabFM, Chronos-2 with covariates) win? | **Refuted.** None beats the random walk or GARCH. |
| R-006 | Can anything beat HAR-RV at forecasting realized volatility? | **Refuted.** LightGBM vs HAR is not significant. |
| R-007 / R-008 | Does a GVZ-implied range (raw or risk-premium scaled) beat GARCH? | **No.** GVZ forecasts the size of the move better, but its ranges are not better. GARCH stays the range engine. |
| R-009 | Does a vol forecast predict large moves? | **Refuted.** It does no better than the plain event frequency. |
| R-010 | Is direction predictable when the model is confident? | **Refuted.** Confident calls are *worse*, below 50%. |
| R-011 | Do valuation features (real rates, dollar, price vs its 5-year mean) work at 3 to 12 months? | **Refuted.** The 12-month ratio is 0.89, but there are only about 11 independent periods, so this cannot be told apart from noise. |
| R-012 | Does pooling gold, silver, platinum and palladium help? | **Refuted.** Worse, especially monthly. |

**Bottom line:** gold returns are close to unpredictable from public data at these horizons. Ranges
are useful and point forecasts are not. The dashboard puts this in front of the user. The ⭐ best model is
only "lowest past error", and the 80% range is the main message.

Full tables: `reports/<study>/report.md`; machine-readable gate verdicts: `reports/gates.json`.

## Quick start

```bash
uv venv .venv && uv pip install -e ".[dev,dashboard]"
python -m goldcast fetch        # download every series -> data/raw/*.parquet + manifest.json
python -m goldcast build        # panel shapes and feature coverage
python -m goldcast backtest     # R-002 walk-forward, both frequencies (~6 min CPU)
python -m goldcast forecast     # next-period forecasts from every model
python -m pytest
```

These commands re-run the studies: `chronos`, `intervals`, `blocks`, `foundation`, `vol`, `tail`,
`selective`, `long`, `pooled`, `range`. Use `python -m goldcast <cmd> --help` to see each one's options.
The foundation-model arms need torch, and the weights download on first use.

## Layout

    configs/default.toml        series catalog, publication lags, windows, test window
    src/goldcast/data/          fetchers (Yahoo, FRED csv, GPR xls) and parquet store + manifest
    src/goldcast/features/      point-in-time panel: values moved to availability date, as-of join
    src/goldcast/models/        baselines, ridge/logit, LightGBM point/quantile/classifier,
                                foundation models, EWMA/GARCH intervals, HAR-RV
    src/goldcast/evaluation/    walk-forward (expanding, purged), DM, PT, pinball, Holm, coverage
    src/goldcast/dashboard/     Streamlit app; horizon.py = 1 week .. 2 year forecast engine
    reports/                    cached predictions, study reports and gate verdicts
    tests/                      leakage (truncation + poisoning), metrics, purging, FM leakage

## Rules the code enforces

- **Publication lags.** Each value enters the panel on the date it became public. Market closes lag
  0 business days, FRED daily series lag 1, and GPR lags 7.
- **Log returns only.** The target is the log return from one decision close to the next, never the
  price level.
- **Purged training rows.** No training target ends after the forecast origin.
- **Train-only fitting.** Imputers and scalers are fit on the training rows only.
- **Shared splits.** Every model uses the same splits, and the baselines are refit every period.

## Extending

- **A series:** add a `[[series]]` row in `configs/default.toml` with its source, lag and kinds.
- **A source:** add a fetcher to `FETCHERS`.
- **A model:** add a class with `fit(X, y)` / `predict(X)` and register it in
  `goldcast/models/__init__.py`.

## Disclaimer

This is research code, not financial advice. In every test here, no model beat the simple baselines
by a statistically significant margin.
