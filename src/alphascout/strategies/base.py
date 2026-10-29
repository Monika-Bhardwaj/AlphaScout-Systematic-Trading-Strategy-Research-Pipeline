"""Strategy contract: weights(panel, cache) -> (T, A) *target* weights decided at the close of
each row using information up to and including that row. The engine alone applies the one-bar
execution lag, so a strategy cannot get timing wrong by itself."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..data.panel import Panel
from ..features.cache import FeatureCache


class Strategy(Protocol):
    family: str

    @property
    def name(self) -> str: ...

    def weights(self, panel: Panel, cache: FeatureCache) -> np.ndarray: ...
