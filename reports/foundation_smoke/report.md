# R-005 foundation arms - verdict: **refuted** (Holm family 16)

## weekly (20 origins, 2015-01-02..2015-05-15; 0 in 2023+)
RW mse 0.000478; lgbm mse 0.000573; lgbm_q pinball 0.005827; GARCH pinball 0.005551; always-up hit 0.500

| arm | mse/rw | mse/rw 2023+ | mse/lgbm | p_holm vs lgbm | p_holm vs rw | pinball | pin/garch | pin/lgbm_q | cov | p_holm vs lgbm_q | p_holm vs garch | hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tabpfn | 1.0166 | - | 0.8478 | 1.0000 | 1.0000 | 0.005533 | 0.9967 | 0.9495 | 0.750 | 0.7999 | 1.0000 | 0.550 |
| chronos2x | 1.0502 | - | 0.8758 | 1.0000 | 1.0000 | 0.005626 | 1.0135 | 0.9655 | 0.750 | 1.0000 | 1.0000 | 0.650 |

## monthly (20 origins, 2015-01-30..2016-08-31; 0 in 2023+)
RW mse 0.002058; lgbm mse 0.003087; lgbm_q pinball 0.013596; GARCH pinball 0.011720; always-up hit 0.500

| arm | mse/rw | mse/rw 2023+ | mse/lgbm | p_holm vs lgbm | p_holm vs rw | pinball | pin/garch | pin/lgbm_q | cov | p_holm vs lgbm_q | p_holm vs garch | hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tabpfn | 1.1197 | - | 0.7465 | 0.4975 | 1.0000 | 0.012272 | 1.0470 | 0.9026 | 0.750 | 1.0000 | 1.0000 | 0.350 |
| chronos2x | 1.0572 | - | 0.7048 | 0.4975 | 1.0000 | 0.011695 | 0.9978 | 0.8602 | 0.850 | 1.0000 | 1.0000 | 0.550 |

## Gates

- weekly:tabpfn:point_vs_lgbm: False
- weekly:tabpfn:point_vs_rw: False
- weekly:tabpfn:quantile_vs_lgbm_q: False
- weekly:tabpfn:quantile_vs_garch: False
- weekly:chronos2x:point_vs_lgbm: False
- weekly:chronos2x:point_vs_rw: False
- weekly:chronos2x:quantile_vs_lgbm_q: False
- weekly:chronos2x:quantile_vs_garch: False
- monthly:tabpfn:point_vs_lgbm: False
- monthly:tabpfn:point_vs_rw: False
- monthly:tabpfn:quantile_vs_lgbm_q: False
- monthly:tabpfn:quantile_vs_garch: False
- monthly:chronos2x:point_vs_lgbm: False
- monthly:chronos2x:point_vs_rw: False
- monthly:chronos2x:quantile_vs_lgbm_q: False
- monthly:chronos2x:quantile_vs_garch: False

Useful: none
