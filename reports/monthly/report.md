# Backtest — monthly

Test window 2015-01-30 .. 2026-08-31 (140 periods; 44 in the 2023+ sub-window).

## Point forecasts (log return)

| model | MSE/rw | DM | p | p Holm | MSE/rw 2023+ | MAE |
|---|---|---|---|---|---|---|
| rw | 1.0000 | — | — | — | 1.0000 | 0.03439 |
| drift | 0.9668 | 1.09 | 0.1390 | 1.0000 | 0.9293 | 0.03403 |
| ses | 0.9807 | 0.34 | 0.3689 | 1.0000 | 0.9350 | 0.03389 |
| ridge | 1.4180 | -3.12 | 0.9989 | 1.0000 | 1.2926 | 0.04100 |
| lgbm | 1.0863 | -1.53 | 0.9360 | 1.0000 | 0.9744 | 0.03657 |

## Direction

| model | hit | always-up hit | PT p | p Holm | hit 2023+ | strat Sharpe | B&H Sharpe | time long |
|---|---|---|---|---|---|---|---|---|
| always_up | 0.557 | 0.557 | — | — | 0.659 | 0.68 | 0.68 | 1.00 |
| logit | 0.536 | 0.557 | 0.2588 | 0.7042 | 0.659 | 0.58 | 0.68 | 0.58 |
| lgbm_clf | 0.564 | 0.557 | 0.1368 | 0.5474 | 0.614 | 0.65 | 0.68 | 0.69 |

## 80% interval (q = 0.1 / 0.5 / 0.9)

| model | mean pinball | pinball 2023+ | coverage | coverage 2023+ | DM p vs qbase |
|---|---|---|---|---|---|
| qbase | 0.010930 | 0.013154 | 0.764 | 0.659 | — |
| lgbm_q | 0.011248 | 0.013677 | 0.721 | 0.568 | 0.8567 |
