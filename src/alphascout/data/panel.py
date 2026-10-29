"""Dense (dates x symbols) NumPy view of the point-in-time store."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Panel:
    dates: np.ndarray                 # datetime64[D], shape (T,)
    symbols: list[str]                # length A
    fields: dict[str, np.ndarray]     # name -> (T, A) float64
    listed: np.ndarray                # (T, A) bool: inside [first bar, last bar] of the security
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def A(self) -> int:
        return len(self.symbols)

    def __getitem__(self, k: str) -> np.ndarray:
        return self.fields[k]

    def has(self, k: str) -> bool:
        return k in self.fields

    def head(self, n: int) -> "Panel":
        """First n dates, as an independent copy (used by the truncation leakage audit)."""
        return Panel(
            self.dates[:n].copy(),
            list(self.symbols),
            {k: v[:n].copy() for k, v in self.fields.items()},
            self.listed[:n].copy(),
            dict(self.meta),
        )


def ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs down axis 0 (causal)."""
    T, A = x.shape
    idx = np.where(np.isnan(x), 0, np.arange(T)[:, None])
    np.maximum.accumulate(idx, axis=0, out=idx)
    return x[idx, np.arange(A)[None, :]]


def select_columns(panel: Panel, cols) -> Panel:
    """Sub-panel on a subset of symbols (used to demonstrate survivorship bias)."""
    cols = np.asarray(cols)
    return Panel(panel.dates, [panel.symbols[i] for i in cols],
                 {k: v[:, cols] for k, v in panel.fields.items()}, panel.listed[:, cols], dict(panel.meta))
