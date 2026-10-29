from collections import Counter

import numpy as np
import pytest

from alphascout.backtest.costs import CostModel
from alphascout.backtest.engine import pnl_vectorized
from alphascout.backtest.event_sim import ExecConfig, simulate
from alphascout.data.pit_store import PITStore
from alphascout.data.synthetic import generate_synthetic, noise_world
from alphascout.features.cache import FeatureCache
from alphascout.risk.metrics import sharpe
from alphascout.robustness.shocks import (adversarial_drawdown, drop_bars, factor_reversal, flash_crash,
                                          gap_on_positions, inject_bad_ticks)
from alphascout.strategies.momentum import CSMomentum


@pytest.fixture(scope="module")
def book():
    P = PITStore(noise_world(60, 900, 7)).build_panel()
    return P, CSMomentum(60, 5, 5, 0.2).weights(P, FeatureCache(P))


def _after(res, P, day, n=40):
    return Counter(e["control"] for e in res.events if str(P.dates[day]) <= e["date"] <= str(P.dates[day + n]))


def test_event_sim_basic_accounting_and_partial_fills(book):
    P, W = book
    r = simulate(P, W, ExecConfig())
    assert np.isfinite(r.equity).all() and r.equity[0] > 0
    assert 0.3 < r.fill_ratio < 1.0                    # participation cap leaves some orders unfilled
    # like-for-like: decide at close t, fill at close t, no frictions -> must reproduce the vectorised engine
    free = simulate(P, W, ExecConfig(fees_bps=0, spread_bps=0, impact_k=0, borrow_bps_annual=0, controls=False,
                                     participation_cap=1e9, delay_bars=0, fill_at="close"))
    v = pnl_vectorized(W, P["ret"], CostModel(0.0, 0.0))[0]
    assert abs(sharpe(free.returns) - sharpe(v)) < 0.1


def test_costs_and_latency_hurt(book):
    P, W = book
    base = simulate(P, W, ExecConfig(controls=False))
    assert simulate(P, W, ExecConfig(controls=False, cost_multiplier=3.0)).equity[-1] < base.equity[-1]
    assert base.cost_total > 0


def test_daily_loss_halt_fires_on_adversarial_gap(book):
    P, W = book
    Ps, hit = gap_on_positions(P, W, 400, -0.20)
    assert len(hit) > 3
    assert _after(simulate(Ps, W, ExecConfig()), P, 400, 1)["daily_loss_halt"] == 1


def test_controls_cut_losses_in_a_sustained_bear_raid(book):
    P, W = book
    Pr = adversarial_drawdown(P, W, 400, 5, -0.15)
    on, off = simulate(Pr, W, ExecConfig()), simulate(Pr, W, ExecConfig().no_controls())
    assert on.equity[420] / on.equity[399] > off.equity[420] / off.equity[399] + 0.08
    assert on.halted_bars > 0 and off.halted_bars == 0


def test_name_freeze_fires_on_untrusted_gap_prints(book):
    P, W = book
    Pf, hit = flash_crash(P, 300, frac=0.1, gap=-0.5, seed=1)
    r = simulate(Pf, W, ExecConfig())
    froze = [e for e in r.events if e["control"] == "name_freeze" and e["date"] == str(P.dates[300])]
    assert len(froze) == len(hit)
    assert not [e for e in simulate(Pf, W, ExecConfig().no_controls()).events if e["control"] == "name_freeze"]


def test_factor_reversal_regime_break_trips_drawdown_halt_and_controls_help(book):
    P, W = book
    Pb = factor_reversal(P, 400, 15, strength=0.02)
    on, off = simulate(Pb, W, ExecConfig()), simulate(Pb, W, ExecConfig().no_controls())
    assert _after(on, P, 400)["drawdown_halt"] >= 1
    assert on.equity[440] / on.equity[399] > off.equity[440] / off.equity[399] + 0.1


def test_shocked_panels_stay_internally_consistent(book):
    P, W = book
    Ps, _ = flash_crash(P, 300, frac=0.2, gap=-0.3, seed=2)
    assert np.allclose(np.diff(Ps["logp"][290:310], axis=0)[np.isfinite(Ps["ret"][291:310])],
                       np.log1p(Ps["ret"][291:310])[np.isfinite(Ps["ret"][291:310])], atol=1e-9)


def test_bad_tick_burst_is_caught_by_validation():
    raw = generate_synthetic(40, 700, seed=9, bad_tick_rate=0.0)
    bad, truth = inject_bad_ticks(raw, n=40, seed=1)
    log = PITStore(bad).rejection_log()
    caught = {(s, d) for s, d in zip(log["symbol"], log["date"])}
    hit = sum((s, d) in caught for s, d in zip(truth["symbol"], truth["date"]))
    assert hit / truth.height >= 0.95


def test_missing_bars_do_not_break_panel_or_strategies():
    raw = drop_bars(generate_synthetic(40, 700, seed=9), frac=0.02, seed=3)
    P = PITStore(raw).build_panel()
    W = CSMomentum(60, 5, 5, 0.2).weights(P, FeatureCache(P))
    assert np.isfinite(W).all() and np.isfinite(pnl_vectorized(W, P["ret"], CostModel())[0]).all()
    assert not np.isinf(P["ret"]).any()
