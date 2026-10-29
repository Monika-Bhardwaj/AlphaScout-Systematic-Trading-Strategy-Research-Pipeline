"""Vectorised PnL engine (screening stage) + a deliberately naive loop reference.

Convention, enforced here and nowhere else: target weights W[t] are decided at the close of row t,
so the position that earns row t's return is W[t-1]. Costs are charged on |W[t]-W[t-1]| in row t."""
from __future__ import annotations

import numpy as np

from .costs import CostModel


def pnl_vectorized(W: np.ndarray, ret: np.ndarray, cost: CostModel, ppy: int = 252):
    Wp = np.zeros_like(W)
    Wp[1:] = W[:-1]
    gross = np.where(np.isfinite(ret), Wp * ret, 0.0).sum(axis=1)
    turnover = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1])))).sum(axis=1)
    short = np.clip(-Wp, 0.0, None).sum(axis=1)
    c = cost.multiplier * (turnover * cost.one_way_bps / 1e4 + short * cost.borrow_bps_annual / 1e4 / ppy)
    return gross - c, gross, turnover


def pnl_reference_loop(W: np.ndarray, ret: np.ndarray, cost: CostModel, ppy: int = 252):
    """Same accounting, written as the per-day / per-asset loop an event-driven simulator uses.
    Exists to (a) prove the vectorised engine is equivalent and (b) time the speedup."""
    T, A = W.shape
    net = np.zeros(T); gross = np.zeros(T); turn = np.zeros(T)
    for t in range(T):
        g = tn = sh = 0.0
        for a in range(A):
            wp = W[t - 1, a] if t > 0 else 0.0
            r = ret[t, a]
            if r == r:
                g += wp * r
            tn += abs(W[t, a] - wp)
            if wp < 0:
                sh += -wp
        gross[t], turn[t] = g, tn
        net[t] = g - cost.multiplier * (tn * cost.one_way_bps / 1e4 + sh * cost.borrow_bps_annual / 1e4 / ppy)
    return net, gross, turn
