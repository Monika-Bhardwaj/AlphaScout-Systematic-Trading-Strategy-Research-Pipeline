from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.panel import Panel
from ..features.cache import FeatureCache
from ..features.cointegration import select_pairs


@dataclass(frozen=True)
class PairsCoint:
    """Gatev-style pairs on Engle-Granger cointegration. Every `hold` rows pairs are re-selected
    from the trailing `formation` rows only; hedge ratio and spread mean/sd are frozen at selection
    time (no peeking at the trading window). Capital 1/k per pair, hedged beta:1 in log space."""
    formation: int = 252
    hold: int = 63
    k: int = 5
    z_entry: float = 2.0
    z_exit: float = 0.5
    p_max: float = 0.05
    family: str = "pairs"

    @property
    def name(self) -> str:
        return f"pairs_F{self.formation}_h{self.hold}_k{self.k}_ze{self.z_entry}_zx{self.z_exit}_p{self.p_max}"

    def weights(self, panel: Panel, cache: FeatureCache) -> np.ndarray:
        logp = panel["logp"]
        T, A = logp.shape
        W = np.zeros((T, A))
        F, H = self.formation, self.hold
        for t0 in range(F, T, H):
            sel = cache.get(("pairs", F, t0, self.p_max),
                            lambda: select_pairs(logp[t0 - F + 1: t0 + 1], k_max=10, p_max=self.p_max))
            pairs = sel[: self.k]
            if not pairs:
                continue
            I = np.array([p.i for p in pairs]); J = np.array([p.j for p in pairs])
            beta = np.array([p.beta for p in pairs]); mu = np.array([p.mu for p in pairs])
            sd = np.array([p.sigma for p in pairs])
            t1 = min(t0 + H, T)
            Z = (logp[t0:t1][:, I] - beta * logp[t0:t1][:, J] - mu) / sd
            pos = np.zeros(len(pairs))
            for r in range(t1 - t0):
                z = Z[r]
                ok = np.isfinite(z) & panel.listed[t0 + r, I] & panel.listed[t0 + r, J]
                leave = ((pos == 1) & (z >= -self.z_exit)) | ((pos == -1) & (z <= self.z_exit)) | ~ok
                pos[leave] = 0
                flat = pos == 0
                pos[flat & ok & (z > self.z_entry)] = -1   # spread rich: short i, long beta*j
                pos[flat & ok & (z < -self.z_entry)] = 1
                W[t0 + r, I] = pos / (1 + beta) / self.k
                W[t0 + r, J] = -pos * beta / (1 + beta) / self.k
        return W
