"""R-005: foundation arms cannot see values after the forecast origin.

The tabular wrappers run through walk_forward, so poisoning every target and feature
after an origin must leave that origin's forecast unchanged. Chronos-2 with covariates
gets the same overwrite-the-future check. These load real weights, so they are slow.
"""

import numpy as np
import pandas as pd
import pytest

from goldcast import fm
from goldcast.evaluation.walkforward import walk_forward

pytestmark = pytest.mark.slow


def _panel(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-04", periods=n, freq="W-FRI")
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=idx,
                     columns=["gold_logret1", "silver_logret1", "vix_diff1", "vix_level"])
    y = pd.Series(rng.normal(0, 0.02, n), index=idx)
    y.iloc[-1] = np.nan  # last target not yet known, as in the real panel
    return X, y


def _poison(X, y, origin):
    Xp, yp = X.copy(), y.copy()
    Xp.loc[Xp.index > origin] = Xp.loc[Xp.index > origin] * 1000 + 7
    # y at the origin is the return to the next period: unknown at the origin
    yp.loc[yp.index >= origin] = yp.loc[yp.index >= origin] * 1000 + 7
    return Xp, yp


@pytest.mark.parametrize("arm", ["tabpfn", "tabfm"])
def test_tabular_arm_ignores_the_future(arm):
    X, y = _panel()
    origin = X.index[70]
    fac = (lambda: fm.TabPFNArm([0.1, 0.5, 0.9])) if arm == "tabpfn" else fm.TabFMArm
    a = walk_forward(fac, X.loc[:origin], y.loc[:origin], str(origin.date()), refit_every=4)
    Xp, yp = _poison(X, y, origin)
    b = walk_forward(fac, Xp, yp, str(origin.date()), refit_every=4)
    np.testing.assert_allclose(np.asarray(b.loc[origin], float), np.asarray(a.loc[origin], float), rtol=1e-5, atol=1e-6)


def test_chronos2x_ignores_the_future():
    from goldcast.models.foundation import load_pipeline

    pipe = load_pipeline(4)
    X, y = _panel()
    origins = X.index[[60, 70]]
    a = fm.chronos2x_forecasts(pipe, X, y, str(origins[0].date()), [0.1, 0.5, 0.9])
    for o in origins:
        Xp, yp = _poison(X, y, o)
        b = fm.chronos2x_forecasts(pipe, Xp, yp, str(origins[0].date()), [0.1, 0.5, 0.9])
        np.testing.assert_allclose(b.loc[o].values, a.loc[o].values, rtol=1e-5)


def test_chronos2x_covariates_cover_the_target_period():
    X, y = _panel()
    cov = fm.chronos_covariates(X)
    assert list(cov.columns) == ["silver_logret1", "vix_diff1"]
    assert cov["silver_logret1"].iloc[5] == X["silver_logret1"].iloc[6]
