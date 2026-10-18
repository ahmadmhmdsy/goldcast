# R-006 realized vol (Holm family 6)

V1 GARCH beats HAR (prior: no): [False, False]; V2 HAR-X beats HAR: [False, False]; V3 LightGBM beats HAR-X: [True, False]

## weekly (615 origins, 2015-01-02..2026-10-09; 197 in 2023+; training from 2008-06-06)

| model | QLIKE | QLIKE/har | QLIKE 2023+ | MSE log RV | cov 80% | cov 2023+ |
|---|---|---|---|---|---|---|
| har | 0.4241 | 1.0000 | 0.3962 | 1.1175 | 0.831 | 0.817 |
| harx | 1630.9883 | 3845.3153 | 136.4560 | 8.7144 | 0.709 | 0.782 |
| lgbm | 0.4098 | 0.9661 | 0.3915 | 0.8498 | 0.780 | 0.777 |
| garch | 0.4149 | 0.9783 | 0.4087 | 1.0486 | 0.826 | 0.822 |
| gvz | 0.3774 | 0.8899 | 0.3535 | 1.0432 | 0.846 | 0.827 |

| test | model vs | p | p_holm | passes |
|---|---|---|---|---|
| V1 | garch vs har | 0.1657 | 0.4971 | False |
| V2 | harx vs har | 1.0000 | 1.0000 | False |
| V3 | lgbm vs harx | 0.0000 | 0.0000 | True |

Range-engine candidates (pass + coverage in [0.75, 0.85]): ['lgbm']

## monthly (140 origins, 2015-01-30..2026-08-31; 44 in 2023+; training from 2008-06-30)

| model | QLIKE | QLIKE/har | QLIKE 2023+ | MSE log RV | cov 80% | cov 2023+ |
|---|---|---|---|---|---|---|
| har | 0.2617 | 1.0000 | 0.2923 | 0.4723 | 0.829 | 0.727 |
| harx | 3265.9391 | 12478.9563 | 0.2694 | 4.5023 | 0.750 | 0.727 |
| lgbm | 0.2377 | 0.9081 | 0.2872 | 0.3979 | 0.793 | 0.705 |
| garch | 0.1966 | 0.7511 | 0.2457 | 0.3680 | 0.821 | 0.750 |
| gvz | 0.1665 | 0.6363 | 0.1903 | 0.3288 | 0.850 | 0.773 |

| test | model vs | p | p_holm | passes |
|---|---|---|---|---|
| V1 | garch vs har | 0.0330 | 0.1570 | False |
| V2 | harx vs har | 0.9686 | 1.0000 | False |
| V3 | lgbm vs harx | 0.0314 | 0.1570 | False |

Range-engine candidates (pass + coverage in [0.75, 0.85]): none

