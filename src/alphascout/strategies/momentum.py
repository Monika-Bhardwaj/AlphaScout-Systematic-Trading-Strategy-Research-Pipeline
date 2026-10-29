from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.panel import Panel
from ..features.cache import FeatureCache
from ..features.core import momentum


@dataclass(frozen=True)
class CSMomentum:
    """Cross-sectional momentum: long top-q, short bottom-q by trailing return, dollar-neutral,
    gross exposure 1, rebalanced every `rebalance` rows."""
    lookback: int = 120
    skip: int = 5
    rebalance: int = 21
    quantile: float = 0.2
    family: str = "momentum"

    @property
    def name(self) -> str:
        return f"mom_L{self.lookback}_s{self.skip}_h{self.rebalance}_q{self.quantile}"

    def weights(self, panel: Panel, cache: FeatureCache) -> np.ndarray:
        m = cache.get(("mom", self.lookback, self.skip), lambda: momentum(panel["logp"], self.lookback, self.skip))
        T, A = m.shape
        elig = panel.listed & np.isfinite(m)
        W = np.zeros((T, A))
        cur = np.zeros(A)
        for t in range(T):
            if t % self.rebalance == 0:
                cur = np.zeros(A)
                e = np.flatnonzero(elig[t])
                if len(e) >= 10:
                    k = max(1, int(len(e) * self.quantile))
                    order = e[np.argsort(m[t, e], kind="stable")]
                    cur[order[-k:] ] = 0.5 / k
                    cur[order[:k]] = -0.5 / k
            else:
                cur = np.where(panel.listed[t], cur, 0.0)
            W[t] = cur
        return W
