"""Rolling Hurst exponent (scaling of lagged-difference dispersion). H<0.5 mean-reverting,
H~0.5 random walk, H>0.5 trending."""
from __future__ import annotations

import numpy as np


def hurst_exponent(x: np.ndarray, lags=(2, 4, 8, 16)) -> float:
    x = np.asarray(x, float)
    sd = np.array([np.std(x[l:] - x[:-l], ddof=1) for l in lags])
    return float(np.polyfit(np.log(lags), np.log(sd), 1)[0])


def rolling_hurst(logp: np.ndarray, window: int = 120, step: int = 5, lags=(2, 4, 8, 16)) -> np.ndarray:
    """H estimated every `step` rows from the trailing `window`, held until the next estimate.
    Row t only ever carries an estimate computed at some t' <= t."""
    T, A = logp.shape
    out = np.full((T, A), np.nan)
    ll = np.log(np.asarray(lags, float))
    llc = ll - ll.mean()
    for t in range(window - 1, T, step):
        seg = logp[t - window + 1: t + 1]
        ok = np.isfinite(seg).all(axis=0)
        sds = np.empty((len(lags), A))
        for k, lag in enumerate(lags):
            sds[k] = (seg[lag:] - seg[:-lag]).std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ls = np.log(sds)
            h = (llc[:, None] * (ls - ls.mean(axis=0))).sum(0) / (llc ** 2).sum()
        h[~ok] = np.nan
        out[t: t + step] = h   # rows t..t+step-1 reuse the estimate known at t
    return out
