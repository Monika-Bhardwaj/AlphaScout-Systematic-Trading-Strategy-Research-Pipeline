import zipfile
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from alphascout.data.binance import parse_klines
from alphascout.data.nse_bhavcopy import NSEBhavcopyAdapter, parse_bhavcopy
from alphascout.data.panel import Panel
from alphascout.data.pit_store import PITStore
from alphascout.features.core import volume_imbalance


def _write_udiff(raw, root, syms, zipped_every=3):
    df = raw.prices.filter(pl.col("symbol").is_in(syms))
    for k, (d, g) in enumerate(df.group_by("date", maintain_order=True)):
        out = pl.DataFrame({
            "TradDt": [d[0].isoformat()] * g.height, "BizDt": [d[0].isoformat()] * g.height,
            "TckrSymb": g["symbol"], "SctySrs": ["EQ"] * g.height, "ISIN": ["IN" + s for s in g["symbol"]],
            "OpnPric": g["open"], "HghPric": g["high"], "LwPric": g["low"], "ClsPric": g["close"],
            "TtlTradgVol": g["volume"], "TtlTrfVal": g["value"]})
        # a non-EQ row that must be filtered out
        junk = out.head(1).with_columns(pl.lit("BE").alias("SctySrs"), pl.lit("JUNKBE").alias("TckrSymb"))
        csv = pl.concat([out, junk]).write_csv()
        p = root / f"BhavCopy_NSE_CM_0_0_0_{d[0]:%Y%m%d}_F_0000.csv"
        if k % zipped_every == 0:
            with zipfile.ZipFile(str(p) + ".zip", "w") as z:
                z.writestr(p.name, csv)
        else:
            p.write_text(csv)


def test_nse_udiff_roundtrip_filters_series_zip_and_keys_by_isin(raw, tmp_path):
    syms = ["SYM001", "SYM002", "SYM003"]
    _write_udiff(raw, tmp_path, syms)
    rd = NSEBhavcopyAdapter(str(tmp_path), key="isin").load()
    src = raw.prices.filter(pl.col("symbol").is_in(syms))
    assert rd.prices.height == src.height                      # BE rows dropped, zip + plain files both read
    assert set(rd.prices["symbol"]) == {"IN" + s for s in syms}
    a = rd.prices.sort(["symbol", "date"])["close"].to_numpy()
    b = src.with_columns(("IN" + pl.col("symbol")).alias("symbol")).sort(["symbol", "date"])["close"].to_numpy()
    assert np.allclose(a, b)
    assert (rd.prices["as_of"].dt.date() == rd.prices["date"]).all() and rd.periods_per_year == 252
    rd.check_schema()


def test_nse_legacy_layout_and_bad_header(raw):
    g = raw.prices.filter(pl.col("symbol") == "SYM004").head(3)
    leg = pl.DataFrame({"SYMBOL": g["symbol"], "SERIES": ["EQ"] * 3, "OPEN": g["open"], "HIGH": g["high"], "LOW": g["low"],
                        "CLOSE": g["close"], "TOTTRDQTY": g["volume"], "TOTTRDVAL": g["value"],
                        "TIMESTAMP": [f"{d:%d-%b-%Y}".upper() for d in g["date"]], "ISIN": ["INX"] * 3})
    out = parse_bhavcopy(leg, key="symbol")
    assert out["date"].to_list() == g["date"].to_list() and np.allclose(out["close"].to_numpy(), g["close"].to_numpy())
    with pytest.raises(ValueError, match="Unrecognised Bhavcopy header"):
        parse_bhavcopy(pl.DataFrame({"foo": ["1"]}))


def test_nse_delisting_inferred_from_bar_spans(raw, tmp_path):
    dead = [s for s, sv in zip(raw.truth["symbols"], raw.truth["survivor"]) if not sv][:1]
    live = [s for s, sv in zip(raw.truth["symbols"], raw.truth["survivor"]) if sv][:1]
    _write_udiff(raw, tmp_path, dead + live)
    sec = NSEBhavcopyAdapter(str(tmp_path)).load().securities
    row = {r["symbol"]: r for r in sec.iter_rows(named=True)}
    assert row["IN" + dead[0]]["delisting_date"] is not None and row["IN" + live[0]]["delisting_date"] is None


def test_binance_kline_parser_and_true_volume_imbalance():
    rows = [[1704067200000, "42000", "42500", "41800", "42300", "100", 1704153599999, "4200000", 1200, "70", "2900000", "0"],
            [1704153600000, "42300", "42400", "41000", "41200", "120", 1704239999999, "5000000", 1500, "30", "1250000", "0"]]
    df = parse_klines(rows, "BTCUSDT")
    assert df["date"].to_list() == [date(2024, 1, 1), date(2024, 1, 2)]
    assert df["buy_volume"].to_list() == [70.0, 30.0] and df["value"][0] == 4200000.0
    assert df["as_of"][0] == datetime(2024, 1, 2, 0, 0, 0)          # known only once the bar closes (+1 ms)
    f = {"volume": df["volume"].to_numpy()[:, None], "buy_volume": df["buy_volume"].to_numpy()[:, None]}
    p = Panel(np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"), ["BTCUSDT"], f, np.ones((2, 1), bool))
    vi = volume_imbalance(p, 2)[1, 0]
    assert vi == pytest.approx(((2 * 70 - 100) + (2 * 30 - 120)) / 220)
