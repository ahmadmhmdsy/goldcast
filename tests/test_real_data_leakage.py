"""R-004: the truncated-build test on the real stored series, every block included.

Skipped when `data/raw` has not been fetched (the offline suite stays offline).
"""

import pandas as pd
import pytest

from goldcast.config import load_config
from goldcast.features.build import build_panel

CUTOFFS = ["2009-12-31", "2016-03-16", "2020-07-03", "2024-11-29"]


@pytest.fixture(scope="module")
def real():
    cfg = load_config()
    if not all((cfg.raw_dir / f"{s.name}.parquet").exists() for s in cfg.series):
        pytest.skip("real raw data not fetched")
    raw = {s.name: pd.read_parquet(cfg.raw_dir / f"{s.name}.parquet")["value"] for s in cfg.series}
    return cfg, raw


@pytest.mark.parametrize("freq", ["weekly", "monthly"])
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_real_truncated_build_matches_full(real, freq, cutoff):
    cfg, raw = real
    X_full, _ = build_panel(raw, cfg.series, cfg.freqs[freq], cfg.target)
    X_cut, _ = build_panel(raw, cfg.series, cfg.freqs[freq], cfg.target, cutoff=cutoff)
    common = X_cut.index.intersection(X_full.index)
    assert len(common) == len(X_cut)
    assert any(c.startswith(("cot_", "gtrends", "epu", "move", "stlfsi")) for c in X_cut.columns)
    pd.testing.assert_frame_equal(X_cut.loc[common], X_full.loc[common])
