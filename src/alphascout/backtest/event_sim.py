"""Sequential event-driven simulator for finalists / paper-trading replay.

Unlike the vectorised screen it keeps state day by day: positions in currency, cash, a delayed
order queue (decision at close t, first possible fill at bar t+delay), fills at a chosen reference
price (VWAP by default) with participation caps (partial fills carry over), spread + fees +
square-root market impact, borrow cost, and risk controls that can actually fire:
  - per-name weight cap and gross-leverage cap on targets,
  - name freeze: no trading in a name for `cooldown_bars` after a print moves more than
    `shock_freeze_ret` in one bar (treated as untrusted data),
  - daily-loss halt: flatten and stand down after a daily loss beyond the limit,
  - drawdown halt: same, on peak-to-trough drawdown.
Positions are tracked as notional and grown with total returns, so splits and dividends never touch
fills; intraday P&L comes from same-day ratios (fill price / close), which are split-invariant."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..data.panel import Panel
from ..features.core import roll_mean, roll_std


@dataclass(frozen=True)
class ExecConfig:
    capital: float = 1e8
    delay_bars: int = 1
    fill_at: str = "vwap"            # "vwap" | "open" | "close"
    participation_cap: float = 0.10  # max fraction of (lagged) ADV traded per name per bar
    fees_bps: float = 10.0           # commission + taxes, per unit notional traded (assumption)
    spread_bps: float = 6.0          # full spread; half is paid per trade (assumption)
    impact_k: float = 0.3            # sqrt-law coefficient on daily vol (assumption)
    borrow_bps_annual: float = 100.0
    cost_multiplier: float = 1.0
    adv_window: int = 20
    controls: bool = True
    max_weight: float = 0.20
    max_gross: float = 1.5
    shock_freeze_ret: float = 0.35
    daily_loss_limit: float = 0.05
    drawdown_limit: float = 0.20
    cooldown_bars: int = 5
    drawdown_cooldown_bars: int = 20

    def no_controls(self) -> "ExecConfig":
        return replace(self, controls=False)


@dataclass
class SimResult:
    dates: np.ndarray
    equity: np.ndarray
    returns: np.ndarray
    events: list = field(default_factory=list)
    fill_ratio: float = float("nan")
    traded_notional: float = 0.0
    cost_total: float = 0.0
    halted_bars: int = 0


def _lag1(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[1:] = x[:-1]
    return out


def simulate(panel: Panel, W: np.ndarray, cfg: ExecConfig = ExecConfig(), start: int = 0,
             end: int | None = None, ppy: int = 252) -> SimResult:
    T, A = panel.T, panel.A
    end = T if end is None else end
    close, ret = panel["close"], panel["ret"]
    ref = {"vwap": panel["vwap"], "open": panel["open"], "close": close}[cfg.fill_at]
    value = panel["vwap"] * panel["volume"]
    adv, sd = _lag1(roll_mean(value, cfg.adv_window)), _lag1(roll_std(ret, cfg.adv_window))
    mult = cfg.cost_multiplier
    V = np.zeros(A)
    cash = cfg.capital
    eq_prev = peak = cash
    frozen_until = np.full(A, -10 ** 9)
    halt_until = -1
    eq, events = [], []
    traded = desired_abs = cost_total = 0.0
    halted_bars = 0
    gross_warned = False
    inf = float("inf")
    max_w = cfg.max_weight if cfg.controls else inf
    max_g = cfg.max_gross if cfg.controls else inf

    for t in range(start, end):
        c, r = close[t], ret[t]
        has_bar = np.isfinite(c)
        dead = (~has_bar) & np.isfinite(r) & (V != 0)           # delisting: realise terminal return
        if dead.any():
            cash += float((V[dead] * (1.0 + r[dead])).sum())
            events.append({"date": str(panel.dates[t]), "control": "delisting_liquidation",
                           "detail": [panel.symbols[a] for a in np.flatnonzero(dead)]})
            V[dead] = 0.0
        fp = np.where(has_bar & np.isfinite(ref[t]), ref[t], c)
        ratio = np.where(has_bar, fp / np.where(has_bar, c, 1.0), 1.0)
        V = V * np.where(has_bar & np.isfinite(r), (1.0 + r) * ratio, 1.0)   # close(t-1) -> fill price

        if cfg.controls:
            shock = has_bar & (np.abs(np.nan_to_num(r)) > cfg.shock_freeze_ret)
            for a in np.flatnonzero(shock & (frozen_until < t)):
                events.append({"date": str(panel.dates[t]), "control": "name_freeze",
                               "detail": f"{panel.symbols[a]} moved {r[a]:+.0%} in one bar; trading frozen"})
            frozen_until[shock] = t + cfg.cooldown_bars
        frozen = frozen_until >= t

        equity_pre = cash + V.sum()
        halted = cfg.controls and t <= halt_until
        halted_bars += int(halted)
        td = t - cfg.delay_bars
        tw = np.zeros(A) if (halted or td < 0) else np.clip(W[td], -max_w, max_w)
        gross = np.abs(tw).sum()
        if gross > max_g:
            tw *= max_g / gross
            if not gross_warned:
                events.append({"date": str(panel.dates[t]), "control": "gross_cap", "detail": f"gross {gross:.2f}"})
                gross_warned = True
        ok = has_bar & ~frozen & np.isfinite(adv[t]) & (adv[t] > 0)
        desired = np.where(ok, tw * equity_pre - V, 0.0)
        cap = cfg.participation_cap * np.where(ok, adv[t], 0.0)
        trade = np.clip(desired, -cap, cap)
        with np.errstate(invalid="ignore", divide="ignore"):
            imp = np.where(ok, cfg.impact_k * np.nan_to_num(sd[t]) * np.sqrt(np.abs(trade) / np.where(ok, adv[t], 1.0)), 0.0)
        cost = mult * np.abs(trade) * ((cfg.fees_bps + cfg.spread_bps / 2.0) / 1e4 + imp)
        cash -= float(trade.sum() + cost.sum())
        cash -= mult * float(np.clip(-(V + trade), 0.0, None).sum()) * cfg.borrow_bps_annual / 1e4 / ppy
        V = V + trade
        traded += float(np.abs(trade).sum()); desired_abs += float(np.abs(desired).sum()); cost_total += float(cost.sum())
        V = V * np.where(has_bar, c / fp, 1.0)                   # fill price -> close(t)

        eq_t = cash + float(V.sum())
        if cfg.controls:
            if eq_t / eq_prev - 1.0 < -cfg.daily_loss_limit and not halted:
                halt_until = t + cfg.cooldown_bars
                events.append({"date": str(panel.dates[t]), "control": "daily_loss_halt",
                               "detail": f"{eq_t / eq_prev - 1.0:+.1%} in one bar; flatten + stand down {cfg.cooldown_bars} bars"})
            peak = max(peak, eq_t)
            if eq_t / peak - 1.0 < -cfg.drawdown_limit and not halted:
                halt_until = t + cfg.drawdown_cooldown_bars
                events.append({"date": str(panel.dates[t]), "control": "drawdown_halt",
                               "detail": f"drawdown {eq_t / peak - 1.0:.1%}; flatten + stand down {cfg.drawdown_cooldown_bars} bars"})
                peak = eq_t
        eq.append(eq_t)
        eq_prev = eq_t

    eq = np.array(eq)
    rets = np.diff(np.concatenate([[cfg.capital], eq])) / np.concatenate([[cfg.capital], eq[:-1]])
    return SimResult(panel.dates[start:end], eq, rets, events,
                     traded / desired_abs if desired_abs > 0 else float("nan"), traded, cost_total, halted_bars)
