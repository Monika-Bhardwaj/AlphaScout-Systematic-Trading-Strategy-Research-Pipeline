from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CostModel:
    """Linear screening-stage costs. one_way_bps covers commission + taxes + half-spread +
    slippage per unit of notional traded; borrow_bps_annual is a financing/borrow proxy on short
    notional (as far as I know, shorting cash equities overnight is restricted in India, so treat
    shorts as futures-implemented). All numbers are assumptions to set per venue/instrument, not
    measured values. `multiplier` is the stress knob (2x, 3x)."""
    one_way_bps: float = 15.0
    borrow_bps_annual: float = 100.0
    multiplier: float = 1.0

    def scaled(self, m: float) -> "CostModel":
        return replace(self, multiplier=self.multiplier * m)
