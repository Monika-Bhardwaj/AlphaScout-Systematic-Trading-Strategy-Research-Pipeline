"""Container + schema for what every data adapter must return."""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

PRICE_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "as_of"]
OPTIONAL_PRICE_COLS = ["value", "buy_volume"]
SECURITY_COLS = ["symbol", "listing_date", "delisting_date", "delist_ret"]
ACTION_COLS = ["symbol", "ex_date", "kind", "price_factor", "amount", "announce_date"]


@dataclass
class RawMarketData:
    """prices: one row per (symbol, date) bar; `as_of` is when the bar became knowable.
    securities: listing/delisting dates (+ optional realised delisting return).
    actions: corporate actions. kind in {"split","bonus","dividend"}; price_factor is
    new_price/old_price for splits/bonuses (0.5 for 2-for-1); amount is cash/share for dividends.
    """
    prices: pl.DataFrame
    securities: pl.DataFrame
    actions: pl.DataFrame
    periods_per_year: int = 252
    default_cost_bps: float = 15.0
    name: str = "unnamed"
    truth: dict | None = field(default=None, repr=False)

    def check_schema(self) -> None:
        for df, cols, nm in [(self.prices, PRICE_COLS, "prices"),
                             (self.securities, SECURITY_COLS, "securities"),
                             (self.actions, ACTION_COLS, "actions")]:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"{nm} frame missing columns {missing}; has {df.columns}")
        bad = self.prices.filter(pl.col("as_of").dt.date() < pl.col("date"))
        if bad.height:
            raise ValueError(f"{bad.height} price rows have as_of earlier than their bar date")


def empty_actions() -> pl.DataFrame:
    return pl.DataFrame(schema={"symbol": pl.Utf8, "ex_date": pl.Date, "kind": pl.Utf8,
                                "price_factor": pl.Float64, "amount": pl.Float64,
                                "announce_date": pl.Date})
