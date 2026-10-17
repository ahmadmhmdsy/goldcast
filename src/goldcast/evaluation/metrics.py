"""Forecast-evaluation statistics used by the R-002 gates."""

from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano(loss_base: np.ndarray, loss_model: np.ndarray, h: int = 1) -> tuple[float, float]:
    """One-sided DM test that the model's loss is LOWER than the baseline's.

    HAC variance with h-1 autocovariance lags, Harvey-Leybourne-Newbold small-sample
    correction, Student-t with n-1 df. Returns (statistic, p-value).
    """
    d = np.asarray(loss_base, float) - np.asarray(loss_model, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return np.nan, np.nan
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, h):
        var += 2 * (dc[k:] @ dc[:-k]) / n
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(stats.t.sf(dm, df=n - 1))


def pesaran_timmermann(actual_up: np.ndarray, pred_up: np.ndarray) -> tuple[float, float]:
    """One-sided Pesaran-Timmermann test of directional accuracy. NaN if forecasts never vary."""
    a = np.asarray(actual_up, bool)
    p = np.asarray(pred_up, bool)
    n = a.size
    py, px = a.mean(), p.mean()
    hit = (a == p).mean()
    pstar = py * px + (1 - py) * (1 - px)
    v_hit = pstar * (1 - pstar) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) / n
              + (2 * px - 1) ** 2 * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n**2)
    denom = v_hit - v_star
    if denom <= 0:
        return np.nan, np.nan
    s = (hit - pstar) / np.sqrt(denom)
    return float(s), float(stats.norm.sf(s))


def pinball(y: np.ndarray, qpred: np.ndarray, quantiles) -> np.ndarray:
    """Per-row mean pinball loss across the quantiles; qpred has shape (n, len(quantiles))."""
    y = np.asarray(y, float)[:, None]
    q = np.asarray(quantiles, float)[None, :]
    diff = y - qpred
    return np.mean(np.maximum(q * diff, (q - 1) * diff), axis=1)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values; NaN entries stay NaN and do not count toward m."""
    items = sorted(((p, k) for k, p in pvals.items() if np.isfinite(p)))
    m = len(items)
    out = {k: np.nan for k in pvals}
    running = 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def long_flat(y: np.ndarray, long: np.ndarray, cost_bp: float, periods_per_year: int) -> dict:
    """Long/flat strategy in log-return terms, charging cost_bp on every position change."""
    y = np.asarray(y, float)
    pos = np.asarray(long, float)
    turns = np.abs(np.diff(np.concatenate([[0.0], pos])))
    r = pos * y - turns * cost_bp / 1e4

    def summary(x):
        sd = x.std(ddof=1)
        return {
            "total_logret": float(x.sum()),
            "ann_ret": float(x.mean() * periods_per_year),
            "sharpe": float(x.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan,
        }

    return {"strategy": summary(r), "buy_hold": summary(y), "time_long": float(pos.mean()),
            "trades": int(turns.sum())}
