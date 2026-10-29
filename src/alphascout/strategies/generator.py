"""Systematic candidate generation. N (the number of trials) is fixed here and carried through to
the deflated Sharpe ratio: every generated candidate counts, whether or not it looks good."""
from __future__ import annotations

import itertools

import numpy as np

from .mean_reversion import TSMeanReversion
from .momentum import CSMomentum
from .pairs import PairsCoint


def default_grid() -> list:
    out = []
    for L, s, h, q in itertools.product([60, 120, 250], [5, 20], [5, 21], [0.2, 0.1]):
        out.append(CSMomentum(L, s, h, q))
    for L, ze, zx, hm in itertools.product([10, 20, 40, 60], [1.5, 2.0, 2.5], [0.0, 0.5], [None, 0.45]):
        out.append(TSMeanReversion(L, ze, zx, hm))
    for F, h, k, ze, zx in itertools.product([126, 252], [21, 63], [5, 10], [2.0, 2.5], [0.0, 0.5]):
        out.append(PairsCoint(F, h, k, ze, zx))
    return out


def _random_extra(rng: np.random.Generator):
    fam = rng.choice(["momentum", "mean_reversion", "pairs"])
    if fam == "momentum":
        L = int(rng.choice([40, 60, 90, 120, 180, 250])); s = int(rng.choice([0, 5, 10, 20]))
        return CSMomentum(L, s, int(rng.choice([5, 10, 21, 42])), float(rng.choice([0.1, 0.15, 0.2, 0.3])))
    if fam == "mean_reversion":
        return TSMeanReversion(int(rng.choice([5, 10, 15, 20, 30, 40, 60, 90])),
                               float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])), float(rng.choice([0.0, 0.25, 0.5])),
                               rng.choice([None, 0.4, 0.45, 0.5]))
    return PairsCoint(int(rng.choice([90, 126, 189, 252, 378])), int(rng.choice([10, 21, 42, 63])),
                      int(rng.choice([3, 5, 8, 10])), float(rng.choice([1.5, 2.0, 2.5, 3.0])),
                      float(rng.choice([0.0, 0.25, 0.5])), float(rng.choice([0.01, 0.05])))


def generate_candidates(n: int | None = None, seed: int = 0) -> list:
    """Deterministic grid (104 candidates). If n < grid: seeded subsample; if n > grid: grid plus
    seeded random draws from wider ranges, de-duplicated."""
    grid = default_grid()
    rng = np.random.default_rng(seed)
    if n is None or n == len(grid):
        return grid
    if n < len(grid):
        idx = np.sort(rng.choice(len(grid), size=n, replace=False))
        return [grid[i] for i in idx]
    seen = {g.name for g in grid}
    out = list(grid)
    while len(out) < n:
        c = _random_extra(rng)
        if c.name not in seen:
            seen.add(c.name)
            out.append(c)
    return out
