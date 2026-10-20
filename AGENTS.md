# gold_forecasting — scoped entry

## What this component is

**gold_forecasting** (package `goldcast`) — a point-in-time gold forecasting
pipeline. It uses free Yahoo, FRED and GPR data, a purged walk-forward, and
DM, PT and pinball gates against naive baselines. It is a git repository of its
own (local only, no remote). Its pre-registrations are R-002 to R-012 in
`research/experiments/`.

## Entry points

| entry point | path | notes |
|---|---|---|
| CLI | `src/goldcast/__main__.py` | `python -m goldcast fetch / build / backtest / forecast / chronos / intervals ...` |
| config | `configs/default.toml` | data windows and model settings |
| dashboard | `src/goldcast/dashboard/app.py`, `run_dashboard.bat` | Streamlit, port 8501 |
| results | `reports/<study>/gates.json`, `reports/<study>/report.md` | the ledger cites these |
| leakage tests | `tests/test_no_leakage.py`, `tests/test_fm_leakage.py`, `tests/test_real_data_leakage.py` | guard the point-in-time property |

## Smallest validation command

From `projects/gold_forecasting/`:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected result: 35 tests, all passing. `test_tabular_arm_ignores_the_future[tabpfn]`
can fail by about 2e-6 against an atol of 1e-6; that is TabPFN numeric noise,
not a leak. Rerun it before treating it as one.

## Boundaries

Owns the data fetchers, features, models, walk-forward and gate code, and the
reports. Hypotheses, gates and verdicts live in `research/experiments/` and
`research/LEDGER.md`, not here. Downloaded data in `data/` is the project's own
cache.

## Hazards

- The venv was built at `E:\python_projects\gold_forecasting`. The editable
  install and `activate` scripts were repointed on 2026-10-20, but `.exe`
  launchers in `.venv\Scripts` (for example `pytest.exe`) can still carry the old
  path. Use `python -m <tool>`.
- Chronos-2, TabPFN and TabFM arms download weights (hundreds of MB) on first
  use and are slow on CPU. A new download needs the owner's OK.
- One GPU job at a time; the foundation-model arms can take the GPU.
- `reports/` holds the numbers the ledger cites; regenerate into a new folder
  rather than over a cited one.

## Pointers to root rules

Root policy lives in `rules/`, registered by `rules/00-index.md`; this file
restates none of it. This component's own rule file is
`rules/components/gold_forecasting.md`; the shared project rules are
`rules/components/projects.md`. Lessons: `docs/memory/gold_forecasting.md`.
