# Backtest — weekly

Test window 2015-01-02 .. 2026-10-09 (615 periods; 197 in the 2023+ sub-window).

## Point forecasts (log return)

| model | MSE/rw | DM | p | p Holm | MSE/rw 2023+ | MAE |
|---|---|---|---|---|---|---|
| rw | 1.0000 | — | — | — | 1.0000 | 0.01658 |
| chronos | 0.9982 | 0.17 | 0.4306 | 1.0000 | 0.9829 | 0.01648 |

## Direction

| model | hit | always-up hit | PT p | p Holm | hit 2023+ | strat Sharpe | B&H Sharpe | time long |
|---|---|---|---|---|---|---|---|---|
| always_up | 0.553 | 0.553 | — | — | 0.589 | 0.67 | 0.67 | 1.00 |
| chronos | 0.553 | 0.553 | — | — | 0.589 | 0.67 | 0.67 | 1.00 |

## 80% interval (q = 0.1 / 0.5 / 0.9)

| model | mean pinball | pinball 2023+ | coverage | coverage 2023+ | DM p vs qbase |
|---|---|---|---|---|---|
| qbase | 0.005405 | 0.006326 | 0.780 | 0.721 | — |
| chronos | 0.005345 | 0.006203 | 0.807 | 0.756 | 0.0677 |
