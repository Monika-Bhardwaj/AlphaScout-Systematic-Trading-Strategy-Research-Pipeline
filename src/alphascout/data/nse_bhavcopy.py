"""NSE cash-market Bhavcopy adapter (reads files you have already downloaded).

I could not download from NSE in the build sandbox, so this adapter was tested only against fixtures
written from the documented column layouts (tests/test_adapters.py), NOT against real NSE files.
Both layouts are supported and auto-detected from the header:
  legacy (pre July 2024):  SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY, TOTTRDVAL, TIMESTAMP, ISIN
  UDiFF  (current):        TckrSymb, SctySrs, OpnPric, HghPric, LwPric, ClsPric, TtlTradgVol, TtlTrfVal, TradDt, ISIN
If your files differ, the adapter raises and prints the header it found; edit the column maps below.

What the files do NOT contain, and what this adapter does about it:
  * corporate actions  -> pass `actions_csv` (symbol, ex_date, kind[split|bonus|dividend], price_factor,
    amount, [announce_date]). Without it every split is quarantined by the validator and dividends are
    missing from returns (you get price returns).
  * delisting dates/returns -> inferred: a security whose last bar is more than `inactive_days` before
    the end of the data is treated as delisted the day after its last bar; delist_ret is unknown (None),
    and the pipeline applies `default_delist_ret`.
  * renames -> key="isin" (default) follows the ISIN, so a ticker change is not a delisting + IPO.
Short selling cash-segment delivery is restricted in India as far as I know, so long/short results
should be read as futures-implemented; cost parameters are assumptions."""
from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from .raw import RawMarketData, empty_actions

LEGACY = dict(symbol="SYMBOL", series="SERIES", open="OPEN", high="HIGH", low="LOW", close="CLOSE",
              volume="TOTTRDQTY", value="TOTTRDVAL", date="TIMESTAMP", isin="ISIN")
UDIFF = dict(symbol="TckrSymb", series="SctySrs", open="OpnPric", high="HghPric", low="LwPric", close="ClsPric",
             volume="TtlTradgVol", value="TtlTrfVal", date="TradDt", isin="ISIN")


def _read_one(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            raw = z.read(name)
        return pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
    return pl.read_csv(path, infer_schema_length=0)


def parse_bhavcopy(df: pl.DataFrame, series=("EQ",), key: str = "isin", publish_hour: int = 19) -> pl.DataFrame:
    df = df.rename({c: c.strip() for c in df.columns})
    cmap = UDIFF if UDIFF["close"] in df.columns else LEGACY if LEGACY["close"] in df.columns else None
    if cmap is None:
        raise ValueError(f"Unrecognised Bhavcopy header: {df.columns}. Edit LEGACY/UDIFF in nse_bhavcopy.py.")
    missing = [v for k, v in cmap.items() if v not in df.columns and k not in ("isin", "value")]
    if missing:
        raise ValueError(f"Bhavcopy header lacks {missing}; found {df.columns}")
    if series:
        df = df.filter(pl.col(cmap["series"]).str.strip_chars().is_in(list(series)))
    datecol = pl.col(cmap["date"]).str.strip_chars()
    d = (datecol.str.to_date("%Y-%m-%d", strict=False) if cmap is UDIFF
         else datecol.str.to_titlecase().str.to_date("%d-%b-%Y", strict=False))
    ident = pl.col(cmap["isin"]).str.strip_chars() if (key == "isin" and cmap["isin"] in df.columns) \
        else pl.col(cmap["symbol"]).str.strip_chars()
    out = df.select(
        ident.alias("symbol"), d.alias("date"),
        *[pl.col(cmap[k]).cast(pl.Float64, strict=False).alias(k) for k in ("open", "high", "low", "close", "volume")],
        (pl.col(cmap["value"]).cast(pl.Float64, strict=False) if cmap["value"] in df.columns else pl.lit(None, pl.Float64)).alias("value"),
    ).drop_nulls(["date", "symbol"])
    return out.with_columns((pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=publish_hour)).alias("as_of"))


class NSEBhavcopyAdapter:
    name = "nse_bhavcopy"

    def __init__(self, root: str, series=("EQ",), key: str = "isin", actions_csv: str | None = None,
                 inactive_days: int = 30, publish_hour: int = 19, default_cost_bps: float = 15.0):
        self.root, self.series, self.key = Path(root), series, key
        self.actions_csv, self.inactive_days = actions_csv, inactive_days
        self.publish_hour, self.default_cost_bps = publish_hour, default_cost_bps

    def load(self) -> RawMarketData:
        files = sorted([*self.root.rglob("*.csv"), *self.root.rglob("*.zip")])
        if not files:
            raise FileNotFoundError(f"no .csv/.zip Bhavcopy files under {self.root}")
        prices = pl.concat([parse_bhavcopy(_read_one(f), self.series, self.key, self.publish_hour) for f in files],
                           how="vertical_relaxed").sort(["symbol", "date"])
        return build_raw(prices, self.actions_csv, self.inactive_days, 252, self.default_cost_bps, self.name)


def build_raw(prices: pl.DataFrame, actions_csv, inactive_days: int, ppy: int, cost_bps: float, name: str) -> RawMarketData:
    end = prices["date"].max()
    spans = prices.group_by("symbol").agg(pl.col("date").min().alias("listing_date"), pl.col("date").max().alias("last"))
    sec = spans.with_columns(
        pl.when(pl.col("last") < end - timedelta(days=inactive_days)).then(pl.col("last") + pl.duration(days=1))
          .otherwise(None).cast(pl.Date).alias("delisting_date"),
        pl.lit(None, pl.Float64).alias("delist_ret")).select("symbol", "listing_date", "delisting_date", "delist_ret")
    if actions_csv:
        a = pl.read_csv(actions_csv, try_parse_dates=True)
        if "announce_date" not in a.columns:   # assumption: announced the day before the ex-date
            a = a.with_columns((pl.col("ex_date") - pl.duration(days=1)).alias("announce_date"))
        a = a.select("symbol", "ex_date", "kind", "price_factor", "amount", "announce_date").with_columns(
            pl.col("price_factor").cast(pl.Float64), pl.col("amount").cast(pl.Float64))
    else:
        a = empty_actions()
    return RawMarketData(prices=prices, securities=sec, actions=a, periods_per_year=ppy,
                         default_cost_bps=cost_bps, name=name)
