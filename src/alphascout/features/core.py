"""Causal feature primitives on (T, A) arrays. Every value at row t uses rows <= t only."""
from __future__ import annotations

import numpy as np

from ..data.panel import Panel


def roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing-window sum ending at t. NaN until the window is full or if any NaN is inside."""
    v = np.isfinite(x)
    cs = np.cumsum(np.where(v, x, 0.0), axis=0)
    cn = np.cumsum(v, axis=0)
    s, n = cs.copy(), cn.copy()
    s[w:] = cs[w:] - cs[:-w]
    n[w:] = cn[w:] - cn[:-w]
    s[n < w] = np.nan
    return s


def roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    return roll_sum(x, w) / w


def roll_std(x: np.ndarray, w: int) -> np.ndarray:
    s1, s2 = roll_sum(x, w), roll_sum(x * x, w)
    var = (s2 - s1 * s1 / w) / (w - 1)
    return np.sqrt(np.clip(var, 0.0, None))


def momentum(logp: np.ndarray, lookback: int, skip: int) -> np.ndarray:
    """log P[t-skip] - log P[t-lookback]."""
    assert 0 <= skip < lookback
    out = np.full_like(logp, np.nan)
    out[lookback:] = logp[lookback - skip: logp.shape[0] - skip] - logp[: logp.shape[0] - lookback]
    return out


def zscore(logp: np.ndarray, w: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = roll_std(logp, w)
        return np.where(sd > 1e-12, (logp - roll_mean(logp, w)) / sd, np.nan)


def realized_vol(ret: np.ndarray, w: int) -> np.ndarray:
    return roll_std(ret, w)


def volume_imbalance(panel: Panel, w: int) -> np.ndarray:
    """Signed-volume share in [-1, 1]. Uses true taker-buy volume when the adapter supplies it
    (crypto); otherwise the close-location-value proxy, which is all daily bars allow."""
    vol = panel["volume"]
    if panel.has("buy_volume"):
        signed = 2.0 * panel["buy_volume"] - vol
    else:
        h, lo, c = panel["high"], panel["low"], panel["close"]
        rng = h - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            clv = np.where(rng > 0, ((c - lo) - (h - c)) / rng, 0.0)
        signed = clv * vol
    with np.errstate(invalid="ignore", divide="ignore"):
        tot = roll_sum(vol, w)
        return np.where(tot > 0, roll_sum(signed, w) / tot, np.nan)


def vwap_deviation(panel: Panel) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return panel["close"] / panel["vwap"] - 1.0
