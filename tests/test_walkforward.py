import numpy as np
import pandas as pd

from goldcast.evaluation.walkforward import walk_forward


class Spy:
    """Records the last training date of every fit and forecasts it back as an ordinal."""

    cheap = False
    fits: list = []

    def fit(self, X, y):
        self.last_ = X.index.max()
        Spy.fits.append((self.last_, len(X)))
        return self

    def predict(self, X):
        return np.full(len(X), self.last_.toordinal(), float)


def _data(n=60):
    idx = pd.date_range("2014-01-03", periods=n, freq="W-FRI")
    return pd.DataFrame({"x": np.arange(n, dtype=float)}, index=idx), pd.Series(np.arange(n, dtype=float), index=idx)


def test_training_targets_end_before_the_origin():
    X, y = _data()
    Spy.fits = []
    p = walk_forward(Spy, X, y, "2014-06-01", refit_every=4)
    for origin, last_train in zip(p.index, p.values):
        assert pd.Timestamp.fromordinal(int(last_train)) < origin


def test_refit_cadence_and_expanding_window():
    X, y = _data()
    Spy.fits = []
    p = walk_forward(Spy, X, y, "2014-06-01", refit_every=4)
    assert len(Spy.fits) == int(np.ceil(len(p) / 4))
    sizes = [n for _, n in Spy.fits]
    assert sizes == sorted(sizes) and sizes[1] - sizes[0] == 4


def test_rows_without_target_are_not_forecast():
    X, y = _data()
    y.iloc[-1] = np.nan
    Spy.fits = []
    p = walk_forward(Spy, X, y, "2014-06-01", refit_every=4)
    assert X.index[-1] not in p.index
