# Backtest — monthly

Test window 2015-01-30 .. 2026-08-31 (140 periods; 44 in the 2023+ sub-window).

## Point forecasts (log return)

| model | MSE/rw | DM | p | p Holm | MSE/rw 2023+ | MAE |
|---|---|---|---|---|---|---|
| rw | 1.0000 | — | — | — | 1.0000 | 0.03439 |
| chronos | 0.9803 | 0.53 | 0.2991 | 1.0000 | 0.9418 | 0.03414 |

## Direction

| model | hit | always-up hit | PT p | p Holm | hit 2023+ | strat Sharpe | B&H Sharpe | time long |
|---|---|---|---|---|---|---|---|---|
| always_up | 0.557 | 0.557 | — | — | 0.659 | 0.68 | 0.68 | 1.00 |
| chronos | 0.550 | 0.557 | 0.8154 | 1.0000 | 0.659 | 0.67 | 0.68 | 0.99 |

## 80% interval (q = 0.1 / 0.5 / 0.9)

| model | mean pinball | pinball 2023+ | coverage | coverage 2023+ | DM p vs qbase |
|---|---|---|---|---|---|
| qbase | 0.010930 | 0.013154 | 0.764 | 0.659 | — |
| chronos | 0.010934 | 0.013080 | 0.757 | 0.659 | 0.5121 |
