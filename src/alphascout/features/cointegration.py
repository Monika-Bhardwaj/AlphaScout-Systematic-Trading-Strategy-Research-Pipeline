"""Vectorised Engle-Granger scan over all asset pairs in a formation window.

Step 1 OLS of log P_i on log P_j; step 2 ADF(1 lag, no constant) on the residual, p-values from
MacKinnon's cointegration surface (same procedure as statsmodels.coint(maxlag=1, autolag=None);
tests/test_features.py checks agreement). With V assets this tests V(V-1)/2 pairs, so selection
inside the window is itself a multiple-testing exercise; pairs are re-tested out of sample and
every pair-strategy variant counts toward N in the deflated-Sharpe correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.adfvalues import mackinnonp


@dataclass
class Pair:
    i: int
    j: int
    beta: float
    mu: float
    sigma: float
    tstat: float
    pvalue: float
    half_life: float


def _eg_chunk(X: np.ndarray, I: np.ndarray, J: np.ndarray):
    ly, lx = X[:, I], X[:, J]
    mx, my = lx.mean(0), ly.mean(0)
    cxx = ((lx - mx) ** 2).sum(0)
    beta = ((lx - mx) * (ly - my)).sum(0) / cxx
    e = ly - (my - beta * mx) - beta * lx
    de = np.diff(e, axis=0)
    y, x1, x2 = de[1:], e[1:-1], de[:-1]
    s11, s12, s22 = (x1 * x1).sum(0), (x1 * x2).sum(0), (x2 * x2).sum(0)
    sy1, sy2 = (x1 * y).sum(0), (x2 * y).sum(0)
    det = s11 * s22 - s12 ** 2
    b1 = (s22 * sy1 - s12 * sy2) / det
    b2 = (s11 * sy2 - s12 * sy1) / det
    res = y - b1 * x1 - b2 * x2
    sig2 = (res ** 2).sum(0) / (y.shape[0] - 2)
    t = b1 / np.sqrt(sig2 * s22 / det)
    spread = ly - beta * lx
    return beta, t, b1, spread.mean(0), spread.std(0, ddof=1)


def select_pairs(window: np.ndarray, k_max: int = 10, p_max: float = 0.05,
                 hl_range=(2.0, 60.0), chunk: int = 2500) -> list[Pair]:
    """window: (F, A) log prices. Assets with any NaN in the window are skipped.
    Returns up to k_max pairs, best ADF statistic first, each asset used at most once."""
    F, A = window.shape
    valid = np.flatnonzero(np.isfinite(window).all(axis=0))
    if len(valid) < 4 or F < 60:
        return []
    X = window[:, valid]
    iu, ju = np.triu_indices(len(valid), k=1)
    betas, ts, b1s, mus, sds = [], [], [], [], []
    for s in range(0, len(iu), chunk):
        b, t, b1, mu, sd = _eg_chunk(X, iu[s:s + chunk], ju[s:s + chunk])
        betas.append(b); ts.append(t); b1s.append(b1); mus.append(mu); sds.append(sd)
    beta, tstat, b1, mu, sd = map(np.concatenate, (betas, ts, b1s, mus, sds))
    with np.errstate(invalid="ignore", divide="ignore"):
        hl = np.where((b1 > -1) & (b1 < 0), -np.log(2) / np.log1p(b1), np.inf)
    ok = np.isfinite(tstat) & (tstat < -2.8) & (hl >= hl_range[0]) & (hl <= hl_range[1]) & (sd > 1e-6) & (beta > 0)
    out, used = [], set()
    for k in np.flatnonzero(ok)[np.argsort(tstat[ok])]:
        p = float(mackinnonp(tstat[k], regression="c", N=2))
        if p > p_max:
            break
        a, b = int(valid[iu[k]]), int(valid[ju[k]])
        if a in used or b in used:
            continue
        used.update((a, b))
        out.append(Pair(a, b, float(beta[k]), float(mu[k]), float(sd[k]), float(tstat[k]), p, float(hl[k])))
        if len(out) >= k_max:
            break
    return out
