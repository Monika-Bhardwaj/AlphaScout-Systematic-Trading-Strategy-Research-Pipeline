from __future__ import annotations

from typing import Callable

from ..data.panel import Panel


class FeatureCache:
    """Memoises features per panel so N candidate strategies share expensive computations."""

    def __init__(self, panel: Panel):
        self.panel = panel
        self._c: dict = {}

    def get(self, key, fn: Callable):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]
