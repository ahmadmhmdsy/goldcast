"""Volatility interval baselines for R-003. They ignore X and use only the training targets.

Both forecast quantiles of the next return as mean + sigma_next * z_q. GARCH takes
z_q from its fitted Student-t; EWMA takes it from the training residuals y / sigma
(filtered historical simulation).
"""

from __future__ import annotations

import warnings

import numpy as np


class EWMAQuantile:
    cheap = True

    def __init__(self, quantiles, lam: float = 0.94, burn: int = 20):
        self.quantiles, self.lam, self.burn = quantiles, lam, burn

    def fit(self, X, y):
        y = np.asarray(y, float)
        var = np.empty(len(y) + 1)
        var[0] = y[: self.burn].var()
        for t, r in enumerate(y):
            var[t + 1] = self.lam * var[t] + (1 - self.lam) * r * r
        z = y[self.burn:] / np.sqrt(var[self.burn:-1])
        self.q_ = np.sqrt(var[-1]) * np.quantile(z, self.quantiles)
        return self

    def predict(self, X):
        return np.tile(self.q_, (len(X), 1))


class GARCHQuantile:
    """GARCH(1,1), Student-t errors, constant mean; returns are scaled by 100 for the optimiser."""

    cheap = True

    def __init__(self, quantiles):
        self.quantiles = quantiles

    def fit(self, X, y):
        from arch import arch_model

        am = arch_model(100 * np.asarray(y, float), mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
        f = res.forecast(horizon=1, reindex=False)
        mu, var = f.mean.values[-1, 0], f.variance.values[-1, 0]
        z = am.distribution.ppf(np.asarray(self.quantiles), res.params[["nu"]].values)
        self.q_ = (mu + np.sqrt(var) * z) / 100
        return self

    def predict(self, X):
        return np.tile(self.q_, (len(X), 1))
