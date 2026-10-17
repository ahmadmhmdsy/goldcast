"""Chronos-2 zero-shot forecasts on the target's own return history (R-002 dated note).

No fitting: at each origin i the context is y[0 .. i-1], the targets already known
at i (the same purge as `walk_forward` with h = 1). All origins go through the
model in batches; the run is CPU-only so it does not compete with GPU jobs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_ID = "amazon/chronos-2"


def load_pipeline(threads: int | None = None):
    import torch
    from chronos import Chronos2Pipeline

    if threads:
        torch.set_num_threads(threads)
    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")


def chronos_forecasts(pipe, y: pd.Series, test_start: str, quantiles, limit: int | None = None,
                      batch_size: int = 64) -> pd.DataFrame:
    """Quantile forecasts (columns = quantiles) for every test row with a known target."""
    rows = np.flatnonzero((y.index >= pd.Timestamp(test_start)) & y.notna().values)
    if limit:
        rows = rows[:limit]
    contexts = [y.iloc[:i].dropna().to_numpy(np.float32) for i in rows]
    out = []
    for k in range(0, len(contexts), batch_size):
        q, _ = pipe.predict_quantiles(contexts[k:k + batch_size], prediction_length=1,
                                      quantile_levels=list(quantiles))
        out += [t[0, 0, :].float().numpy() for t in q]
    return pd.DataFrame(np.sort(np.array(out), axis=1), index=y.index[rows], columns=list(quantiles))
