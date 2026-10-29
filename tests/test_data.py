import numpy as np
import polars as pl

from alphascout.data.panel import select_columns
from alphascout.data.pit_store import PITStore
from alphascout.data.synthetic import generate_synthetic


def _explained(store, panel):
    di = {d: i for i, d in enumerate(panel.dates)}
    sy = {s: i for i, s in enumerate(panel.symbols)}
    ex = np.zeros((panel.T, panel.A), bool)
    log = store.rejection_log()
    for s, d in zip(log["symbol"], log["date"]):
        ex[di[np.datetime64(d)]: di[np.datetime64(d)] + 4, sy[s]] = True
    return ex


def test_returns_match_ground_truth_except_around_rejections(raw, store, panel):
    """Splits + dividends handled from raw prices; only bars next to a logged rejection may differ."""
    tr = raw.truth["ret"]
    diff = ~np.isclose(tr, panel["ret"], atol=1e-9, equal_nan=True)
    assert (diff & ~_explained(store, panel)).sum() == 0


def test_bad_ticks_and_unadjusted_splits_are_caught(raw, store):
    log = store.rejection_log()
    caught = {(s, d) for s, d in zip(log["symbol"], log["date"])}
    bt = raw.truth["bad_ticks"]
    missed = [(s, d) for s, d in zip(bt["symbol"], bt["date"]) if (s, d) not in caught]
    assert not missed, f"bad ticks not in rejection log: {missed}"
    for s, d, _ in raw.truth["unrecorded_splits"]:
        assert (s, d) in caught


def test_universe_is_survivorship_free_and_matches_snapshot(raw, store):
    t = raw.truth
    for k in (100, 400, 800):
        d = t["dates"][k]
        expected = {t["symbols"][a] for a in np.flatnonzero(t["listed"][k])}
        rep = store.verify_universe_snapshot(d.astype(object), expected)
        assert rep["ok"], rep
    assert (~t["survivor"]).sum() > 0   # the fixture really contains delisted names


def test_survivor_only_universe_is_upward_biased():
    r = generate_synthetic(120, 1000, seed=5, delist_frac=0.4)
    st = PITStore(r); p = st.build_panel()
    surv = np.flatnonzero(r.truth["survivor"])
    def ew(pp):
        # every finite return cell, including the terminal delisting return realised the day after the last bar
        return np.nanmean(pp["ret"])
    assert ew(select_columns(p, surv)) > ew(p)   # dropping delisted names flatters returns


def test_as_of_panel_is_a_prefix_of_the_full_panel(raw, store, panel):
    k = 500
    cut = panel.dates[k].astype("datetime64[D]").astype(object)
    from datetime import datetime, time
    snap = store.build_panel(as_of=datetime.combine(cut, time(23, 59)))
    assert snap.T == k + 1
    cols = [panel.symbols.index(s) for s in snap.symbols]
    a, b = panel["ret"][: k, :][:, cols], snap["ret"][:k]   # last visible bar can't be confirmed, skip it
    di = {d: i for i, d in enumerate(panel.dates)}
    assert np.allclose(a, b, atol=1e-12, equal_nan=True)


def test_future_rows_never_visible_before_as_of(raw, store):
    from datetime import datetime, time
    d = raw.truth["dates"][300].astype(object)
    p, a = store._visible(datetime.combine(d, time(12, 0)))   # before the 19:00 publication of day 300
    assert p["date"].max() < d
    assert a["ex_date"].max() <= d


def test_schema_rejects_lookahead_as_of(raw):
    import pytest
    bad = raw.prices.with_columns((pl.col("as_of") - pl.duration(days=3)).alias("as_of"))
    from dataclasses import replace
    with pytest.raises(ValueError):
        replace(raw, prices=bad).check_schema()
