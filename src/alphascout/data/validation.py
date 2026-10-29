"""Data validation + rejection log (bad ticks, inconsistent bars, unadjusted corporate actions)."""
from __future__ import annotations

import math

import polars as pl

# Daily moves this close to a plausible split/consolidation ratio, with no recorded corporate
# action nearby, are treated as an unadjusted action: the *return* is quarantined (set to NaN)
# instead of being fed to strategies as a fake crash/spike. Ratios milder than 2:1 are left
# alone on purpose: genuine large moves exist and quarantining them would hide real losses.
# The tolerance must exceed a normal day's real return, because an unadjusted split shows up as
# canonical_ratio * (1 + that day's true return): a 3% band missed a 5:1 split that coincided with
# a -6% day (found by the ground-truth test). 12% trades a small false-quarantine risk (a genuine
# ~-50% move with no action) for much better recall.
CANON_RATIOS = [1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2.0, 3.0, 5.0, 10.0]
LOG_SCHEMA = {"symbol": pl.Utf8, "date": pl.Date, "rule": pl.Utf8, "action": pl.Utf8, "detail": pl.Utf8}


def _log(df: pl.DataFrame, rule: str, action: str, detail: str) -> pl.DataFrame:
    return df.select("symbol", "date").with_columns(
        pl.lit(rule).alias("rule"), pl.lit(action).alias("action"), pl.lit(detail).alias("detail"))


def validate_prices(
    prices: pl.DataFrame,
    actions: pl.DataFrame | None = None,
    *,
    spike_ret: float = 0.30,
    spike_revert_tol: float = 0.10,
    split_tol: float = 0.12,
    action_window: str = "3d",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (clean_prices_with_suspect_ret_flag, rejection_log).

    Note on timing: the spike-and-revert rule confirms a bad print using the *next* bar. That
    is fine for an offline backtest but a live system would have to hold the bar in quarantine
    for one day. PITStore.build_panel(as_of=...) re-runs validation on the visible slice, so the
    last visible bar is never "confirmed" with data from its future.
    """
    logs: list[pl.DataFrame] = []
    df = prices.sort(["symbol", "date", "as_of"])

    uniq = df.unique(subset=["symbol", "date"], keep="last", maintain_order=True)
    if uniq.height < df.height:
        removed = df.join(uniq, on=["symbol", "date", "as_of"], how="anti")
        logs.append(_log(removed, "duplicate_bar", "dropped", "kept latest as_of"))
    df = uniq

    eps = 1e-9
    o, h, lo, c, v = (pl.col(x) for x in ["open", "high", "low", "close", "volume"])
    nonpos = pl.any_horizontal(o.is_null(), h.is_null(), lo.is_null(), c.is_null(),
                               o <= 0, h <= 0, lo <= 0, c <= 0)
    incons = ((h < lo * (1 - eps)) | (c > h * (1 + eps)) | (c < lo * (1 - eps))
              | (o > h * (1 + eps)) | (o < lo * (1 - eps)))
    rule = (pl.when(nonpos).then(pl.lit("nonpositive_or_null_price"))
              .when(v < 0).then(pl.lit("negative_volume"))
              .when(incons).then(pl.lit("ohlc_inconsistent"))
              .otherwise(pl.lit(None, dtype=pl.Utf8)))
    df = df.with_columns(rule.alias("_rule"))
    for r in df.filter(pl.col("_rule").is_not_null())["_rule"].unique().to_list():
        logs.append(_log(df.filter(pl.col("_rule") == r), r, "rejected", "bar removed"))
    df = df.filter(pl.col("_rule").is_null()).drop("_rule").sort(["symbol", "date"])

    prev = pl.col("close").shift(1).over("symbol")
    nxt = pl.col("close").shift(-1).over("symbol")
    spike = (((pl.col("close") / prev).log().abs() > math.log(1 + spike_ret))
             & ((nxt / prev).log().abs() < spike_revert_tol))
    df = df.with_columns(spike.fill_null(False).alias("_spike"))
    sp = df.filter(pl.col("_spike"))
    if sp.height:
        logs.append(_log(sp, "bad_tick_spike_revert", "rejected",
                         f">{spike_ret:.0%} one-bar move that reverts within {spike_revert_tol:.0%}"))
    df = df.filter(~pl.col("_spike")).drop("_spike").sort(["symbol", "date"])

    df = df.with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol")).alias("_ratio"))
    near = pl.any_horizontal(*[((pl.col("_ratio") / r) - 1).abs() < split_tol for r in CANON_RATIOS])
    cand = df.filter(near.fill_null(False)).select("symbol", "date", "_ratio")
    suspect_keys = pl.DataFrame(schema={"symbol": pl.Utf8, "date": pl.Date})
    if cand.height:
        if actions is not None and actions.height:
            a = (actions.filter(pl.col("kind").is_in(["split", "bonus"]))
                 .select("symbol", pl.col("ex_date")).sort("ex_date"))
            j = cand.sort("date").join_asof(a, left_on="date", right_on="ex_date", by="symbol",
                                            strategy="nearest", tolerance=action_window)
            unrec = j.filter(pl.col("ex_date").is_null())
        else:
            unrec = cand
        if unrec.height:
            logs.append(_log(unrec, "suspect_unadjusted_action", "quarantine_return",
                             "price ratio matches a split/consolidation, no recorded action within 3d"))
            suspect_keys = unrec.select("symbol", "date")
    df = (df.drop("_ratio")
            .join(suspect_keys.with_columns(pl.lit(True).alias("suspect_ret")), on=["symbol", "date"], how="left")
            .with_columns(pl.col("suspect_ret").fill_null(False)))
    log = pl.concat(logs, how="vertical_relaxed") if logs else pl.DataFrame(schema=LOG_SCHEMA)
    return df, log.sort(["rule", "symbol", "date"]) if log.height else log
