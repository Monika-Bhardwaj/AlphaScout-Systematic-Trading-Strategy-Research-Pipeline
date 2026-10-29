import numpy as np

from alphascout.backtest.costs import CostModel
from alphascout.backtest.engine import pnl_reference_loop, pnl_vectorized
from alphascout.features.cache import FeatureCache
from alphascout.features.leakage import truncation_audit
from alphascout.risk.metrics import sharpe
from alphascout.strategies.generator import default_grid, generate_candidates
from alphascout.strategies.mean_reversion import TSMeanReversion
from alphascout.strategies.momentum import CSMomentum
from alphascout.strategies.pairs import PairsCoint


def test_every_strategy_family_is_causal(panel):
    reps = [CSMomentum(60, 5, 5, 0.2), TSMeanReversion(20, 2.0, 0.0, 0.45), PairsCoint(126, 21, 5, 2.0, 0.5)]
    for s in reps:
        rep = truncation_audit(lambda p, s=s: s.weights(p, FeatureCache(p)), panel, s.name)
        assert not rep.leaked, str(rep)


def test_engine_matches_naive_loop_and_exposure_is_bounded(panel):
    cost = CostModel()
    for s in (CSMomentum(60, 5, 5, 0.2), TSMeanReversion(20, 1.5, 0.0, None), PairsCoint(126, 21, 5, 2.0, 0.5)):
        W = s.weights(panel, FeatureCache(panel))
        assert np.isfinite(W).all() and np.abs(W).sum(1).max() <= 1.0 + 1e-9
        a, *_ = pnl_vectorized(W, panel["ret"], cost)
        b, *_ = pnl_reference_loop(W, panel["ret"], cost)
        assert np.allclose(a, b, atol=1e-12)


def test_engine_enforces_one_bar_execution_lag(panel):
    r = np.nan_to_num(panel["ret"])
    cost = CostModel(0.0, 0.0)
    same_row = np.sign(r) / panel.A            # decided with row t info, earns row t+1: no edge
    foresight = np.zeros_like(r); foresight[:-1] = np.sign(r[1:]) / panel.A   # peeks at t+1: huge "edge"
    assert abs(sharpe(pnl_vectorized(same_row, r, cost)[0])) < 2.0
    assert sharpe(pnl_vectorized(foresight, r, cost)[0]) > 20


def test_costs_scale_linearly_and_generator_counts():
    assert len(default_grid()) == 104 == len(generate_candidates())
    assert len(generate_candidates(30, 1)) == 30 and len(generate_candidates(150, 1)) == 150
    assert len({c.name for c in generate_candidates(150, 1)}) == 150
