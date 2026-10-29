"""Point-in-time (PIT) data store.

Every price row carries an explicit `as_of` timestamp (when it became knowable); every corporate
action carries an `announce_date`. `build_panel(as_of=t)` sees only what was knowable at t, runs
validation on that visible slice, and builds returns from *raw* prices plus actions on their
ex-dates. Returns are never computed from back-adjusted price levels, so future splits cannot leak
into past prices. The universe is whatever had a bar on each date (delisted names included).
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl

from .panel import Panel, ffill
from .raw import RawMarketData
from .validation import LOG_SCHEMA, validate_prices


class PITStore:
    def __init__(self, raw: RawMarketData, validate: bool = True):
        raw.check_schema()
        self.raw = raw
        self.validate = validate
        self.prices = raw.prices
        self.securities = raw.securities
        self.actions = raw.actions
        self._cache: dict = {}

    # ---- visibility -------------------------------------------------------------------
    def _visible(self, as_of: datetime | None):
        if as_of is None:
            return self.prices, self.actions
        d = as_of.date()
        p = self.prices.filter(pl.col("as_of") <= as_of)
        a = self.actions.filter((pl.col("announce_date") <= d) & (pl.col("ex_date") <= d))
        return p, a

    def _clean(self, as_of: datetime | None):
        key = as_of
        if key in self._cache:
            return self._cache[key]
        p, a = self._visible(as_of)
        if self.validate:
            p, log = validate_prices(p, a)
        else:
            p, log = p.with_columns(pl.lit(False).alias("suspect_ret")), pl.DataFrame(schema=LOG_SCHEMA)
        self._cache[key] = (p, a, log)
        return self._cache[key]

    def rejection_log(self, as_of: datetime | None = None) -> pl.DataFrame:
        return self._clean(as_of)[2]

    # ---- universe ---------------------------------------------------------------------
    def universe(self, d: date, as_of: datetime | None = None) -> list[str]:
        p, _ = self._visible(as_of)
        return sorted(p.filter(pl.col("date") == d)["symbol"].to_list())

    def verify_universe_snapshot(self, d: date, expected: set[str], as_of: datetime | None = None) -> dict:
        got = set(self.universe(d, as_of))
        return {"missing": sorted(expected - got), "unexpected": sorted(got - expected),
                "ok": got == set(expected), "n_expected": len(expected), "n_got": len(got)}

    # ---- panel ------------------------------------------------------------------------
    def build_panel(self, as_of: datetime | None = None, default_delist_ret: float = 0.0) -> Panel:
        p, acts, log = self._clean(as_of)
        dates = np.sort(p["date"].unique().to_numpy()).astype("datetime64[D]")
        symbols = sorted(p["symbol"].unique().to_list())
        T, A = len(dates), len(symbols)
        sidx = {s: i for i, s in enumerate(symbols)}
        ti = np.searchsorted(dates, p["date"].to_numpy().astype("datetime64[D]"))
        ai = np.fromiter((sidx[s] for s in p["symbol"].to_list()), dtype=np.int64, count=p.height)

        def wide(col: str) -> np.ndarray:
            out = np.full((T, A), np.nan)
            out[ti, ai] = p[col].to_numpy()
            return out

        fields = {k: wide(k) for k in ["open", "high", "low", "close", "volume"]}
        suspect = np.zeros((T, A), bool)
        suspect[ti, ai] = p["suspect_ret"].to_numpy()
        close = fields["close"]

        F, D = np.ones((T, A)), np.zeros((T, A))
        if acts.height:
            r = np.searchsorted(dates, acts["ex_date"].to_numpy().astype("datetime64[D]"))
            ok = (r < T) & (dates[np.minimum(r, T - 1)] == acts["ex_date"].to_numpy().astype("datetime64[D]"))
            for row, sym, kind, pf, amt in zip(r[ok], np.array(acts["symbol"].to_list())[ok],
                                               np.array(acts["kind"].to_list())[ok],
                                               acts["price_factor"].to_numpy()[ok], acts["amount"].to_numpy()[ok]):
                if sym not in sidx:
                    continue
                if kind == "dividend":
                    D[row, sidx[sym]] += amt
                else:
                    F[row, sidx[sym]] *= pf

        prev = np.vstack([np.full((1, A), np.nan), ffill(close)[:-1]])
        ret = (close / F + D) / prev - 1.0
        ret[suspect] = np.nan

        valid = np.isfinite(close)
        first = valid.argmax(axis=0)
        last = T - 1 - valid[::-1].argmax(axis=0)
        tt = np.arange(T)[:, None]
        listed = (tt >= first[None, :]) & (tt <= last[None, :])

        # delisting returns: realised on the first date after the last bar
        sec = self.securities
        for sym, dl_date, dl_ret in zip(sec["symbol"].to_list(), sec["delisting_date"].to_list(),
                                        sec["delist_ret"].to_list()):
            if sym not in sidx or dl_date is None:
                continue
            a = sidx[sym]
            nxt = last[a] + 1
            if nxt < T and np.datetime64(dl_date) <= dates[-1] and dates[nxt] >= np.datetime64(dl_date):
                ret[nxt, a] = default_delist_ret if dl_ret is None else dl_ret

        lr = np.where(listed, np.nan_to_num(np.log1p(np.clip(ret, -0.999999, None))), 0.0)
        logp = np.where(listed, np.cumsum(lr, axis=0), np.nan)
        fields["ret"] = ret
        fields["logp"] = logp
        if "value" in p.columns:
            val = wide("value")
            with np.errstate(invalid="ignore", divide="ignore"):
                fields["vwap"] = np.where(fields["volume"] > 0, val / fields["volume"], np.nan)
        else:
            fields["vwap"] = (fields["high"] + fields["low"] + close) / 3.0
        if "buy_volume" in p.columns:
            fields["buy_volume"] = wide("buy_volume")
        return Panel(dates, symbols, fields, listed,
                     {"periods_per_year": self.raw.periods_per_year, "name": self.raw.name,
                      "n_rejections": int(log.height)})
