"""Synthetic shocks for adversarial robustness tests. Panel-level shocks rewrite returns AND the
price levels/log-prices that features see, so strategies react to them exactly as to real data."""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import polars as pl

from ..data.panel import Panel
from ..data.raw import RawMarketData


def _copy(panel: Panel) -> Panel:
    return Panel(panel.dates.copy(), list(panel.symbols), {k: v.copy() for k, v in panel.fields.items()},
                 panel.listed.copy(), dict(panel.meta))


def _apply_return_override(p: Panel, new_ret: np.ndarray, rows: slice) -> None:
    """Replace returns on `rows` (NaN entries in new_ret = leave as is) and propagate the level change."""
    old = p["ret"][rows]
    use = np.isfinite(new_ret) & np.isfinite(old)
    factor = np.where(use, (1.0 + np.where(use, new_ret, 0.0)) / (1.0 + np.where(use, old, 0.0)), 1.0)
    cum = np.ones_like(p["ret"])
    cum[rows] = np.cumprod(factor, axis=0)
    r0 = rows.start
    cum[rows.stop:] = cum[rows.stop - 1]
    p["ret"][rows] = np.where(use, new_ret, old)
    for k in ("close", "open", "high", "low", "vwap"):
        p.fields[k] = p.fields[k] * cum
    p.fields["logp"] = p.fields["logp"] + np.log(cum)


def flash_crash(panel: Panel, day: int, frac: float = 0.3, gap: float = -0.30, seed: int = 0) -> tuple[Panel, np.ndarray]:
    """A random `frac` of listed names gap by `gap` on one day (prices stay lower afterwards)."""
    p = _copy(panel)
    rng = np.random.default_rng(seed)
    elig = np.flatnonzero(p.listed[day] & np.isfinite(p["ret"][day]))
    hit = rng.choice(elig, size=max(1, int(frac * len(elig))), replace=False)
    new = np.full((1, p.A), np.nan)
    new[0, hit] = gap
    _apply_return_override(p, new, slice(day, day + 1))
    return p, hit


def gap_on_positions(panel: Panel, W: np.ndarray, day: int, gap: float, side: str = "long") -> tuple[Panel, np.ndarray]:
    """Adversarial gap on the names the book holds going into `day` (W[day-1] is the live position)."""
    p = _copy(panel)
    held = (W[day - 1] > 0) if side == "long" else (W[day - 1] < 0)
    hit = np.flatnonzero(held & p.listed[day] & np.isfinite(p["ret"][day]))
    new = np.full((1, p.A), np.nan)
    new[0, hit] = gap
    _apply_return_override(p, new, slice(day, day + 1))
    return p, hit


def adversarial_drawdown(panel: Panel, W: np.ndarray, start: int, n_days: int, daily_gap: float) -> Panel:
    """A sustained 'bear raid': on each of n_days the book's long names (held from the prior close)
    fall by daily_gap. Worst case for a long/short book short of changing the book."""
    p = panel
    for d in range(start, start + n_days):
        p, _ = gap_on_positions(p, W, d, daily_gap)
    return p


def correlated_regime_break(panel: Panel, start: int, length: int, common_mu: float = -0.02,
                            common_sd: float = 0.03, idio_scale: float = 2.5, seed: int = 0) -> Panel:
    """Correlations go to ~1 and idiosyncratic vol multiplies: r' = common_t + idio_scale * demeaned r."""
    p = _copy(panel)
    rng = np.random.default_rng(seed)
    rows = slice(start, start + length)
    r = p["ret"][rows]
    dem = r - np.nanmean(r, axis=1, keepdims=True)
    common = rng.normal(common_mu, common_sd, (length, 1))
    new = np.where(np.isfinite(r), np.clip(common + idio_scale * dem, -0.9, 3.0), np.nan)
    _apply_return_override(p, new, rows)
    return p


def inject_bad_ticks(raw: RawMarketData, n: int = 40, seed: int = 0) -> tuple[RawMarketData, pl.DataFrame]:
    """Corrupt n random interior bars (spike x0.1/x5, non-positive close, inconsistent OHLC) in the raw
    price frame; returns (corrupted raw, ground-truth frame of what was corrupted)."""
    rng = np.random.default_rng(seed)
    df = raw.prices.sort(["symbol", "date"]).with_row_index("rid")
    pos = df.with_columns(pl.col("rid").rank().over("symbol").alias("_k"), pl.len().over("symbol").alias("_n"))
    cand = pos.filter((pl.col("_k") > 5) & (pl.col("_k") < pl.col("_n") - 5))["rid"].to_numpy()
    pick = rng.choice(cand, size=n, replace=False)
    kinds = rng.choice(["spike", "nonpositive", "ohlc_inconsistent"], size=n, p=[0.6, 0.2, 0.2])
    close = df["close"].to_numpy().copy(); high = df["high"].to_numpy().copy(); low = df["low"].to_numpy().copy()
    for rid, k in zip(pick, kinds):
        if k == "spike":
            f = float(rng.choice([0.1, 0.2, 5.0, 10.0])); close[rid] *= f
            high[rid] = max(high[rid], close[rid]); low[rid] = min(low[rid], close[rid])
        elif k == "nonpositive":
            close[rid] = -1.0
        else:
            high[rid] = low[rid] * 0.9
    out = df.with_columns(pl.Series("close", close), pl.Series("high", high), pl.Series("low", low)).drop("rid")
    truth = df.filter(pl.col("rid").is_in(pick.tolist())).select("symbol", "date").with_columns(pl.Series("kind", kinds))
    return replace(raw, prices=out), truth


def drop_bars(raw: RawMarketData, frac: float = 0.01, seed: int = 0) -> RawMarketData:
    """Randomly delete a fraction of bars (missing data)."""
    rng = np.random.default_rng(seed)
    keep = rng.random(raw.prices.height) >= frac
    return replace(raw, prices=raw.prices.filter(pl.Series(keep)))


def factor_reversal(panel: Panel, start: int, length: int, strength: float = 0.02, lookback: int = 60,
                    common_mu: float = -0.01) -> Panel:
    """Momentum-crash style regime break: for `length` days, past winners are hit and past losers
    rally by `strength` per cross-sectional standard deviation of trailing return, on top of a common
    selloff. Not tailored to any specific book: it breaks the historical relationship momentum books
    rely on (a dollar-neutral book is insulated from the common move but not from this)."""
    p = _copy(panel)
    rows = slice(start, start + length)
    new = np.full((length, p.A), np.nan)
    for k, t in enumerate(range(start, start + length)):
        trail = p["logp"][t - 1] - p["logp"][t - 1 - lookback]
        ok = np.isfinite(trail) & np.isfinite(p["ret"][t])
        if ok.sum() > 3:
            z = np.zeros(p.A)
            z[ok] = (trail[ok] - trail[ok].mean()) / (trail[ok].std() + 1e-12)
            new[k] = np.where(ok, np.clip(p["ret"][t] - strength * z + common_mu / 4, -0.9, 3.0), np.nan)
    _apply_return_override(p, new, rows)
    return p
