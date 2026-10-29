"""Stationary-bootstrap Reality Check with Romano-Wolf stepdown (White 2000; Romano & Wolf 2005).

Null: every candidate has zero true mean. Each strategy's returns are centred, whole time-rows are
resampled in stationary blocks (so cross-strategy dependence and serial dependence are preserved),
and the distribution of the MAXIMUM Sharpe over the strategies still in play is built. A strategy
is significant only if its Sharpe beats that maximum. Adjusted p-values control the family-wise
error rate across ALL candidates, however correlated or heterogeneous they are."""
from __future__ import annotations

import numpy as np

from .stats import sharpe_cols


def stationary_bootstrap_indices(T: int, B: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    jump = rng.random((B, T)) < 1.0 / mean_block
    jump[:, 0] = True
    starts = rng.integers(0, T, (B, T))
    t = np.arange(T)[None, :]
    last = np.maximum.accumulate(np.where(jump, t, 0), axis=1)
    return (np.take_along_axis(starts, last, axis=1) + (t - last)) % T


def romano_wolf(R: np.ndarray, B: int = 1000, mean_block: float = 20.0, seed: int = 0) -> np.ndarray:
    """Step-down FWER-adjusted p-values for H0_k: Sharpe_k <= 0. NaN for degenerate columns."""
    R = np.asarray(R, float)
    T, N = R.shape
    obs = sharpe_cols(R)
    ok = np.flatnonzero(np.isfinite(obs))
    out = np.full(N, np.nan)
    if len(ok) == 0:
        return out
    X = R[:, ok] - R[:, ok].mean(axis=0)
    idx = stationary_bootstrap_indices(T, B, mean_block, np.random.default_rng(seed))
    boot = np.empty((B, len(ok)))
    for b in range(B):
        xb = X[idx[b]]
        sd = xb.std(axis=0, ddof=1)
        boot[b] = np.where(sd > 1e-14, xb.mean(axis=0) / np.maximum(sd, 1e-14), -np.inf)
    o = obs[ok]
    order = np.argsort(-o)
    adj = np.empty(len(ok))
    prev = 0.0
    for j, k in enumerate(order):
        mx = boot[:, order[j:]].max(axis=1)
        p = (1.0 + float((mx >= o[k]).sum())) / (B + 1.0)
        prev = max(prev, p)
        adj[k] = prev
    out[ok] = adj
    return out
