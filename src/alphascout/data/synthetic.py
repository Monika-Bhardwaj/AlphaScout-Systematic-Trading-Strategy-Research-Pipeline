"""Synthetic equity market with *known ground truth*.

Why it exists: every control in AlphaScout (survivorship handling, corporate-action
adjustment, bad-tick rejection, leakage detection, PBO on noise) needs a world where the
right answer is known. Structure is explicit: `market_drift`, `distress_drift` (pre-delisting decay, a predictable
loser effect that survivor-only universes miss), `planted_momentum` and `planted_pairs`. Setting all
four to zero gives a market with NO tradable edge (see `noise_world()`);
`planted_pairs` / `planted_momentum` add real, known structure for positive controls.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

from .raw import RawMarketData


def _pl_dates(arr: np.ndarray) -> pl.Series:
    return pl.Series(arr.astype("datetime64[D]")).cast(pl.Date)


def generate_synthetic(
    n_assets: int = 80,
    n_days: int = 1500,
    seed: int = 0,
    start: str = "2016-01-01",
    planted_pairs: int = 0,
    planted_momentum: float = 0.0,
    delist_frac: float = 0.15,
    ipo_frac: float = 0.15,
    split_frac: float = 0.15,
    unrecorded_split_frac: float = 0.3,
    bad_tick_rate: float = 0.0004,
    market_drift: float = 0.0003,
    distress_drift: float = 0.0012,
) -> RawMarketData:
    rng = np.random.default_rng(seed)
    T, A = n_days, n_assets
    dates = np.busday_offset(np.datetime64(start), np.arange(T), roll="forward").astype("datetime64[D]")
    symbols = [f"SYM{i:03d}" for i in range(A)]

    # ---- returns: market regime + sector factors + idiosyncratic --------------------------
    u = rng.random(T)
    stress = np.zeros(T, bool)
    s = False
    for t in range(T):
        s = (u[t] < 0.95) if s else (u[t] > 0.98)
        stress[t] = s
    vol_mult = np.where(stress, 2.2, 1.0)
    mkt = rng.normal(market_drift, 0.007, T) * vol_mult
    n_sec = 6
    sec = rng.integers(0, n_sec, A)
    secf = rng.normal(0, 0.005, (T, n_sec))
    beta = rng.uniform(0.6, 1.4, A)
    idio_sd = rng.uniform(0.010, 0.022, A)
    lr = (mkt[:, None] * beta[None, :] + secf[:, sec]
          + rng.normal(0, 1, (T, A)) * idio_sd[None, :] * np.sqrt(vol_mult)[:, None])

    if planted_momentum > 0:  # slowly varying latent drift => past returns predict future returns
        mu = np.zeros((T, A))
        eta = rng.normal(0, 5e-5 * planted_momentum, (T, A))
        for t in range(1, T):
            mu[t] = 0.995 * mu[t - 1] + eta[t]
        lr += mu

    perm = rng.permutation(A)
    pairs = [(int(min(perm[2 * k], perm[2 * k + 1])), int(max(perm[2 * k], perm[2 * k + 1])))
             for k in range(planted_pairs)]
    protected = {a for p in pairs for a in p}
    for i, j in pairs:  # log P_j = bp * log P_i + s_t, s_t an OU process => cointegrated
        bp = rng.uniform(0.8, 1.2)
        sp = np.zeros(T)
        eps = rng.normal(0, 0.010, T)
        for t in range(T):
            sp[t] = (0.88 * sp[t - 1] if t else 0.0) + eps[t]
        ds = np.diff(sp, prepend=0.0)
        lr[:, j] = bp * lr[:, i] + ds

    # ---- lifecycles: IPOs and delistings (survivorship structure) -------------------------
    first = np.zeros(A, int)
    last_excl = np.full(A, T)
    delist_ret = np.full(A, np.nan)
    for a in range(A):
        if a in protected:
            continue
        if rng.random() < ipo_frac:
            first[a] = rng.integers(int(0.1 * T), int(0.6 * T))
        if rng.random() < delist_frac:
            lo, hi = max(first[a] + 300, int(0.4 * T)), int(0.95 * T)
            if lo < hi:
                last_excl[a] = rng.integers(lo, hi)
                if rng.random() < 0.7:  # distressed delisting: drift down, terminal loss
                    lr[last_excl[a] - 120:last_excl[a], a] -= distress_drift
                    delist_ret[a] = rng.uniform(-0.5, -0.05)
                else:                   # benign (merger / voluntary)
                    delist_ret[a] = rng.uniform(0.0, 0.08)
    tt = np.arange(T)[:, None]
    listed = (tt >= first[None, :]) & (tt < last_excl[None, :])

    tr = np.expm1(lr)
    ret_true = np.where(listed, tr, np.nan)
    ret_true[first, np.arange(A)] = np.nan  # no return on the first bar
    for a in range(A):
        if last_excl[a] < T:
            ret_true[last_excl[a], a] = delist_ret[a]

    # ---- corporate actions -> raw (unadjusted) prices -------------------------------------
    divmask = (rng.random((T, A)) < 2 / 252) & listed
    divmask[first, np.arange(A)] = False
    y = rng.uniform(0.004, 0.012, (T, A)) * divmask
    split_ratio = np.ones((T, A))
    recorded, unrecorded = [], []
    for a in range(A):
        if rng.random() < split_frac and last_excl[a] - first[a] > 200:
            ts = int(rng.integers(first[a] + 60, last_excl[a] - 60))
            ratio = float(rng.choice([2, 2, 3, 5, 10]))
            split_ratio[ts, a] = ratio
            divmask[ts, a] = False
            y[ts, a] = 0.0
            (unrecorded if rng.random() < unrecorded_split_frac else recorded).append((a, ts, ratio))
    X = np.empty((T, A))
    X[0] = np.exp(rng.normal(np.log(400), 0.9, A))
    for t in range(1, T):
        X[t] = X[t - 1] * (1.0 + tr[t] - y[t])
    S = np.cumprod(split_ratio, axis=0)
    C = X / S

    # ---- OHLCV --------------------------------------------------------------------------
    Op = C * np.exp(-(0.5 * lr + rng.normal(0, 0.3 * idio_sd, (T, A))))
    Hi = np.maximum(Op, C) * np.exp(np.abs(rng.normal(0, 0.5 * idio_sd, (T, A))))
    Lo = np.minimum(Op, C) * np.exp(-np.abs(rng.normal(0, 0.5 * idio_sd, (T, A))))
    lv = np.zeros((T, A))
    nz = rng.normal(0, 0.3, (T, A))
    for t in range(1, T):
        lv[t] = 0.9 * lv[t - 1] + nz[t]
    base = np.exp(rng.normal(12, 1.0, A))
    Vol = np.round(base * np.exp(lv) * (1 + 1.2 * np.abs(lr) / idio_sd))
    Vwap = (Hi + Lo + C) / 3 * (1 + rng.normal(0, 0.001, (T, A)))

    # ---- inject bad ticks (ground truth recorded) ----------------------------------------
    bad = []
    interior = (tt >= first[None, :] + 2) & (tt < last_excl[None, :] - 2)
    cand = np.argwhere(interior)
    n_bad = int(bad_tick_rate * len(cand))
    for k in rng.choice(len(cand), size=n_bad, replace=False) if n_bad else []:
        t, a = cand[k]
        r = rng.random()
        if r < 0.7:
            f = float(rng.choice([0.1, 0.2, 5.0, 10.0]))
            C[t, a] *= f
            Hi[t, a] = max(Hi[t, a], C[t, a])
            Lo[t, a] = min(Lo[t, a], C[t, a])
            kind = "spike"
        elif r < 0.85:
            C[t, a] = -1.0
            kind = "nonpositive"
        else:
            Hi[t, a] = Lo[t, a] * 0.9
            kind = "ohlc_inconsistent"
        bad.append((symbols[a], dates[t].astype(object), kind))

    # ---- assemble long frames (only listed bars exist) ------------------------------------
    ti, ai = np.nonzero(listed)
    asof = (dates[ti].astype("datetime64[us]") + np.timedelta64(19 * 3600, "s"))
    prices = pl.DataFrame({
        "symbol": pl.Series([symbols[a] for a in ai]),
        "date": _pl_dates(dates[ti]),
        "open": Op[ti, ai], "high": Hi[ti, ai], "low": Lo[ti, ai], "close": C[ti, ai],
        "volume": Vol[ti, ai], "value": (Vwap * Vol)[ti, ai],
        "as_of": pl.Series(asof).cast(pl.Datetime("us")),
    })
    sec_rows = [{
        "symbol": symbols[a], "listing_date": dates[first[a]].astype(object),
        "delisting_date": dates[last_excl[a]].astype(object) if last_excl[a] < T else None,
        "delist_ret": None if np.isnan(delist_ret[a]) else float(delist_ret[a]),
    } for a in range(A)]
    securities = pl.DataFrame(sec_rows, schema={"symbol": pl.Utf8, "listing_date": pl.Date,
                                                "delisting_date": pl.Date, "delist_ret": pl.Float64})
    act_rows = []
    di, da = np.nonzero(divmask)
    for t, a in zip(di, da):
        ex = dates[t].astype(object)
        act_rows.append({"symbol": symbols[a], "ex_date": ex, "kind": "dividend", "price_factor": 1.0,
                         "amount": float(y[t, a] * X[t - 1, a] / S[t, a]),
                         "announce_date": ex - timedelta(days=10)})
    for a, ts, ratio in recorded:
        ex = dates[ts].astype(object)
        act_rows.append({"symbol": symbols[a], "ex_date": ex, "kind": "split", "price_factor": 1.0 / ratio,
                         "amount": 0.0, "announce_date": ex - timedelta(days=10)})
    actions = pl.DataFrame(act_rows, schema={"symbol": pl.Utf8, "ex_date": pl.Date, "kind": pl.Utf8,
                                             "price_factor": pl.Float64, "amount": pl.Float64,
                                             "announce_date": pl.Date})
    truth = {
        "dates": dates, "symbols": symbols, "ret": ret_true, "listed": listed, "stress": stress,
        "survivor": last_excl == T, "pairs": [(symbols[i], symbols[j]) for i, j in pairs],
        "bad_ticks": pl.DataFrame(bad, schema={"symbol": pl.Utf8, "date": pl.Date, "kind": pl.Utf8}, orient="row"),
        "unrecorded_splits": [(symbols[a], dates[ts].astype(object), r) for a, ts, r in unrecorded],
        "recorded_splits": [(symbols[a], dates[ts].astype(object), r) for a, ts, r in recorded],
    }
    return RawMarketData(prices=prices, securities=securities, actions=actions, periods_per_year=252,
                         default_cost_bps=15.0, name=f"synthetic(seed={seed})", truth=truth)


def noise_world(n_assets: int = 60, n_days: int = 1500, seed: int = 0, **kw) -> RawMarketData:
    """Market with no tradable structure: zero market drift, no pre-delisting drift, no planted
    momentum or pairs. Delisting jump risk, splits, dividends, bad ticks and vol regimes remain."""
    return generate_synthetic(n_assets, n_days, seed, planted_pairs=0, planted_momentum=0.0,
                              market_drift=0.0, distress_drift=0.0, **kw)
