"""Model registry. Each task maps a model name to a factory taking (cfg, freq)."""

from __future__ import annotations

from goldcast.models.baselines import AlwaysUp, Drift, RandomWalk, RollingQuantile, SES
from goldcast.models.ml import (
    LGBMDirection,
    LGBMPoint,
    LGBMQuantile,
    LogitDirection,
    RidgeModel,
)

REGISTRY = {
    "point": {
        "rw": lambda cfg, f: RandomWalk(),
        "drift": lambda cfg, f: Drift(),
        "ses": lambda cfg, f: SES(),
        "ridge": lambda cfg, f: RidgeModel(),
        "lgbm": lambda cfg, f: LGBMPoint(),
    },
    "quantile": {
        "qbase": lambda cfg, f: RollingQuantile(cfg.quantiles, f.baseline_quantile_window),
        "lgbm_q": lambda cfg, f: LGBMQuantile(cfg.quantiles),
    },
    "direction": {
        "always_up": lambda cfg, f: AlwaysUp(),
        "logit": lambda cfg, f: LogitDirection(),
        "lgbm_clf": lambda cfg, f: LGBMDirection(),
    },
}

BASELINES = {"point": "rw", "quantile": "qbase", "direction": "always_up"}
