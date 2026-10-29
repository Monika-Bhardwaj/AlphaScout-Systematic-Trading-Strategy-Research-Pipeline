"""Binance spot daily klines adapter (public REST, no key).

Network access was blocked in the build sandbox: `parse_klines` is unit-tested against rows in the
documented kline layout, `fetch_klines` / `BinanceAdapter.load` are untested against the live API.
Kline row: [open_time_ms, open, high, low, close, volume, close_time_ms, quote_volume, n_trades,
taker_buy_base_volume, taker_buy_quote_volume, ignore]. taker_buy_base_volume gives TRUE signed volume,
so volume_imbalance() uses it instead of the daily close-location proxy.
Limits: daily bars only (the Panel is date-indexed; intraday needs a datetime-indexed panel, planned);
no corporate actions; you must pass the symbol universe explicitly, so including pairs that were later
delisted is on you (survivorship control is only as good as that list)."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import polars as pl

from .nse_bhavcopy import build_raw
from .raw import RawMarketData

BASE = "https://api.binance.com/api/v3/klines"


def parse_klines(rows: list, symbol: str) -> pl.DataFrame:
    recs = []
    for r in rows:
        close_time = datetime.fromtimestamp(r[6] / 1000, tz=timezone.utc).replace(tzinfo=None)
        recs.append({"symbol": symbol, "date": datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).date(),
                     "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]),
                     "volume": float(r[5]), "value": float(r[7]), "buy_volume": float(r[9]),
                     "as_of": close_time + timedelta(milliseconds=1)})
    return pl.DataFrame(recs, schema={"symbol": pl.Utf8, "date": pl.Date, "open": pl.Float64, "high": pl.Float64,
                                      "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
                                      "value": pl.Float64, "buy_volume": pl.Float64, "as_of": pl.Datetime("us")})


def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "1d", pause: float = 0.2) -> list:
    out, cur = [], start_ms
    while cur < end_ms:
        q = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1000})
        with urllib.request.urlopen(f"{BASE}?{q}", timeout=30) as resp:
            batch = json.loads(resp.read())
        if not batch:
            break
        out.extend(batch)
        cur = batch[-1][6] + 1
        time.sleep(pause)
    return out


class BinanceAdapter:
    name = "binance_spot_1d"

    def __init__(self, symbols: list[str], start: str, end: str, default_cost_bps: float = 10.0):
        self.symbols, self.start, self.end, self.default_cost_bps = symbols, start, end, default_cost_bps

    def load(self) -> RawMarketData:
        s = int(datetime.fromisoformat(self.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
        e = int(datetime.fromisoformat(self.end).replace(tzinfo=timezone.utc).timestamp() * 1000)
        prices = pl.concat([parse_klines(fetch_klines(sym, s, e), sym) for sym in self.symbols]).sort(["symbol", "date"])
        return build_raw(prices, None, 3, 365, self.default_cost_bps, self.name)
