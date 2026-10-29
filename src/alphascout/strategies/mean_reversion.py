from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.panel import Panel
from ..features.cache import FeatureCache
from ..features.core import zscore
from ..features.hurst import rolling_hurst


@dataclass(frozen=True)
class TSMeanReversion:
    """Per-asset z-score of log price vs its rolling mean. Enter long below -z_entry / short above
    +z_entry, exit on crossing -z_exit / +z_exit. Optional Hurst gate: only trade assets whose
    rolling H is below `hurst_max` (mean-reverting regime). Equal capital per listed asset."""
    lookback: int = 20
    z_entry: float = 2.0
    z_exit: float = 0.0
    hurst_max: float | None = None
    family: str = "mean_reversion"

    @property
    def name(self) -> str:
        h = "none" if self.hurst_max is None else self.hurst_max
        return f"mr_L{self.lookback}_ze{self.z_entry}_zx{self.z_exit}_H{h}"

    def weights(self, panel: Panel, cache: FeatureCache) -> np.ndarray:
        z = cache.get(("z", self.lookback), lambda: zscore(panel["logp"], self.lookback))
        gate = None
        if self.hurst_max is not None:
            H = cache.get(("hurst", 120, 5), lambda: rolling_hurst(panel["logp"], 120, 5))
            gate = H < self.hurst_max
        T, A = z.shape
        W = np.zeros((T, A))
        pos = np.zeros(A)
        for t in range(T):
            zt = z[t]
            ok = panel.listed[t] & np.isfinite(zt)
            if gate is not None:
                ok &= gate[t]
            leave = ((pos == 1) & (zt >= -self.z_exit)) | ((pos == -1) & (zt <= self.z_exit)) | ~ok
            pos[leave] = 0
            flat = pos == 0
            pos[flat & ok & (zt > self.z_entry)] = -1
            pos[flat & ok & (zt < -self.z_entry)] = 1
            W[t] = pos
        n_listed = np.maximum(panel.listed.sum(axis=1), 1)[:, None]
        return W / n_listed
