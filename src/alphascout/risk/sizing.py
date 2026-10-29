from __future__ import annotations

import numpy as np

from ..features.core import roll_std


def vol_target(r: np.ndarray, gross: np.ndarray | None = None, target_vol: float = 0.10,
               lookback: int = 63, ppy: int = 252, max_leverage: float = 3.0,
               rescale_cost_bps: float = 0.0):
    """Scale a return stream to a target annual vol. Leverage on row t uses the trailing vol
    through row t-1. Rescaling the book is not free: if `gross` (book gross exposure per row) is
    given, |d leverage| * gross * rescale_cost_bps is charged."""
    T = len(r)
    sd = roll_std(r[:, None], lookback)[:, 0] * np.sqrt(ppy)
    est = np.full(T, np.nan)
    est[1:] = sd[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        lev = np.where(np.isfinite(est) & (est > 0), np.clip(target_vol / est, 0.0, max_leverage), 0.0)
    out = lev * r
    if gross is not None and rescale_cost_bps > 0:
        dlev = np.abs(np.diff(lev, prepend=0.0))
        gp = np.concatenate([[0.0], gross[:-1]])
        out = out - dlev * gp * rescale_cost_bps / 1e4
    return out, lev
