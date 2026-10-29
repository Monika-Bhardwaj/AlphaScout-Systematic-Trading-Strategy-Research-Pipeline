"""Causal market-regime labels for conditional performance breakdowns. The label used for row t
is computed from rows <= t-1, i.e. it is knowable before the return it conditions."""
from __future__ import annotations

import numpy as np

from ..features.core import roll_std
from .metrics import sharpe


def market_regimes(ret: np.ndarray, listed: np.ndarray, vol_window: int = 63, trend_window: int = 126) -> dict:
    T = ret.shape[0]
    x = np.where(listed & np.isfinite(ret), ret, np.nan)
    cnt = np.isfinite(x).sum(axis=1)
    mkt = np.where(cnt > 0, np.nansum(x, axis=1) / np.maximum(cnt, 1), 0.0)
    vol = roll_std(mkt[:, None], vol_window)[:, 0]
    warm = max(vol_window, trend_window) + 1
    high = np.zeros(T, bool)
    up = np.zeros(T, bool)
    valid = np.zeros(T, bool)
    cum = np.cumsum(mkt)
    for t in range(warm, T):
        hist = vol[vol_window - 1: t]            # vol known through t-1
        high[t] = vol[t - 1] > np.nanmedian(hist)
        up[t] = (cum[t - 1] - cum[t - 1 - trend_window]) > 0
        valid[t] = True
    return {"high_vol": high, "uptrend": up, "valid": valid}


def regime_breakdown(r: np.ndarray, reg: dict, ppy: int = 252, min_obs: int = 60) -> dict:
    out = {}
    for label, mask in [("vol=low", ~reg["high_vol"]), ("vol=high", reg["high_vol"]),
                        ("trend=down", ~reg["uptrend"]), ("trend=up", reg["uptrend"])]:
        m = mask & reg["valid"] & np.isfinite(r)
        out[label] = {"n": int(m.sum()), "sharpe": sharpe(r[m], ppy) if m.sum() >= min_obs else None,
                      "mean_daily": float(r[m].mean()) if m.sum() else None}
    return out
