"""Feature-based models. Every preprocessing step is fit on the training rows only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Small, conservative trees: the signal-to-noise ratio of gold returns is tiny.
LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=7,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=1,
    verbose=-1,
)


def _linear_prep():
    return [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]


class RidgeModel:
    cheap = False

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.m_ = make_pipeline(*_linear_prep(), Ridge(alpha=self.alpha)).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X)


class LGBMPoint:
    cheap = False

    def fit(self, X, y):
        self.m_ = LGBMRegressor(**LGBM_PARAMS).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X)


class LGBMQuantile:
    cheap = False

    def __init__(self, quantiles: tuple[float, ...]):
        self.quantiles = quantiles

    def fit(self, X, y):
        self.ms_ = [
            LGBMRegressor(objective="quantile", alpha=q, **LGBM_PARAMS).fit(X, y)
            for q in self.quantiles
        ]
        return self

    def predict(self, X):
        out = np.column_stack([m.predict(X) for m in self.ms_])
        return np.sort(out, axis=1)  # no crossing quantiles


class LogitDirection:
    cheap = False

    def __init__(self, C: float = 0.1):
        self.C = C

    def fit(self, X, y):
        up = (np.asarray(y) > 0).astype(int)
        self.m_ = make_pipeline(*_linear_prep(), LogisticRegression(C=self.C, max_iter=1000)).fit(X, up)
        return self

    def predict(self, X):
        return self.m_.predict_proba(X)[:, 1]


class LGBMDirection:
    cheap = False

    def fit(self, X, y):
        up = (np.asarray(y) > 0).astype(int)
        self.m_ = LGBMClassifier(**LGBM_PARAMS).fit(X, up)
        return self

    def predict(self, X):
        return self.m_.predict_proba(X)[:, 1]
