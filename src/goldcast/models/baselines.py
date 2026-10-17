"""Naive baselines. They ignore X and use only the training targets.

`cheap = True` lets the walk-forward refit them every period instead of every
`refit_every` periods, so a baseline is never handicapped by a stale fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RandomWalk:
    """Zero log-return forecast: tomorrow's price equals today's."""

    cheap = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomWalk":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class Drift:
    """Expanding mean of past returns."""

    cheap = True

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


class SES:
    """Simple exponential smoothing of returns (EWMA level)."""

    cheap = True

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def fit(self, X, y):
        self.level_ = float(pd.Series(np.asarray(y)).ewm(alpha=self.alpha, adjust=False).mean().iloc[-1])
        return self

    def predict(self, X):
        return np.full(len(X), self.level_)


class RollingQuantile:
    """Unconditional quantiles of the last `window` training returns."""

    cheap = True

    def __init__(self, quantiles: tuple[float, ...], window: int):
        self.quantiles = quantiles
        self.window = window

    def fit(self, X, y):
        tail = np.asarray(y)[-self.window:]
        self.q_ = np.quantile(tail, self.quantiles)
        return self

    def predict(self, X):
        return np.tile(self.q_, (len(X), 1))


class AlwaysUp:
    """Direction baseline: probability of an up period is always 1."""

    cheap = True

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X))
