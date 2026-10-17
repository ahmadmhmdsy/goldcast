"""Point-in-time check: building from data truncated at a cutoff must reproduce the full build."""

import numpy as np
import pandas as pd
import pytest

from goldcast.features.build import build_panel

CUTOFFS = ["2008-03-19", "2009-12-31", "2010-07-02", "2011-05-17"]  # mid-week, month end, Friday


@pytest.mark.parametrize("freq", ["weekly", "monthly"])
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_truncated_build_matches_full(synthetic, freq, cutoff):
    raw, catalog, freqs = synthetic
    X_full, y_full = build_panel(raw, catalog, freqs[freq], "gold")
    X_cut, y_cut = build_panel(raw, catalog, freqs[freq], "gold", cutoff=cutoff)

    assert X_cut.index.max() <= pd.Timestamp(cutoff)
    common = X_cut.index.intersection(X_full.index)
    assert len(common) == len(X_cut), "truncated build produced a decision date the full build lacks"
    pd.testing.assert_frame_equal(X_cut.loc[common], X_full.loc[common])

    both = y_cut.notna()
    pd.testing.assert_series_equal(y_cut[both], y_full.loc[y_cut.index][both])


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_future_values_cannot_move_past_features(synthetic, cutoff):
    """Corrupt every observation after the cutoff; nothing on or before the cutoff may change."""
    raw, catalog, freqs = synthetic
    c = pd.Timestamp(cutoff)
    poisoned = {k: v.where(v.index <= c, v * 1000 + 7) for k, v in raw.items()}
    X_a, _ = build_panel(raw, catalog, freqs["weekly"], "gold")
    X_b, _ = build_panel(poisoned, catalog, freqs["weekly"], "gold")
    # Lagged series reach the panel later than their observation date, so only
    # decision dates on or before the cutoff are guaranteed equal.
    early = X_a.index[X_a.index <= c]
    pd.testing.assert_frame_equal(X_a.loc[early], X_b.loc[early])


def test_publication_lag_is_applied(synthetic):
    """A 1-business-day lag means a Friday decision sees Thursday's rate, not Friday's."""
    raw, catalog, freqs = synthetic
    X, _ = build_panel(raw, catalog, freqs["weekly"], "gold")
    d = X.index[50]
    assert d.dayofweek == 4
    assert X.loc[d, "rate_level"] == raw["rate"].loc[d - pd.offsets.BDay(1)]


def test_incomplete_final_period_dropped(synthetic):
    raw, catalog, freqs = synthetic
    X, y = build_panel(raw, catalog, freqs["monthly"], "gold", cutoff="2010-07-14")
    assert X.index.max() == pd.Timestamp("2010-06-30")
    assert np.isnan(y.iloc[-1])


def test_negative_prices_give_nan_not_error(synthetic):
    raw, catalog, freqs = synthetic
    raw = dict(raw, oil=raw["oil"] - raw["oil"].max() - 1)  # entirely negative
    X, _ = build_panel(raw, catalog, freqs["weekly"], "gold")
    assert X.filter(like="oil_logret").isna().all().all()
