"""Automated look-ahead detector (truncation test).

For any function f(panel) -> array: compute f on the full panel and on the panel truncated to its
first n dates. If f is causal, f(full)[:n] == f(truncated) exactly. Any dependence on future rows
(shift(-1), full-sample normalisation, centred windows, global ranks, ...) makes the two differ,
and the report names the first row where they diverge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..data.panel import Panel


@dataclass
class LeakageReport:
    name: str
    leaked: bool
    cuts: list = field(default_factory=list)  # dicts: n, first_bad_row, max_abs_diff

    def __str__(self):
        if not self.leaked:
            return f"[OK]   {self.name}: causal at {len(self.cuts)} cut points"
        c = next(c for c in self.cuts if c["first_bad_row"] is not None)
        return (f"[LEAK] {self.name}: uses future data (first divergent row {c['first_bad_row']} "
                f"when truncated at {c['n']}, max |diff|={c['max_abs_diff']:.3g})")


def truncation_audit(fn: Callable[[Panel], np.ndarray], panel: Panel, name: str = "feature",
                     cut_fracs=(0.45, 0.65, 0.85), atol: float = 1e-10) -> LeakageReport:
    full = np.asarray(fn(panel))
    rep = LeakageReport(name, False)
    for f in cut_fracs:
        n = int(f * panel.T)
        sub = np.asarray(fn(panel.head(n)))
        a = full[:n]
        same = np.isclose(a, sub, atol=atol, rtol=1e-9, equal_nan=True)
        if same.all():
            rep.cuts.append({"n": n, "first_bad_row": None, "max_abs_diff": 0.0})
        else:
            bad_rows = np.flatnonzero(~same.reshape(n, -1).all(axis=1))
            d = np.abs(np.nan_to_num(a) - np.nan_to_num(sub))
            rep.cuts.append({"n": n, "first_bad_row": int(bad_rows[0]), "max_abs_diff": float(d.max())})
            rep.leaked = True
    return rep
