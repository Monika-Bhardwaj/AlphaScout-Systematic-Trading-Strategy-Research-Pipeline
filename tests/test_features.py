import numpy as np
from statsmodels.tsa.stattools import coint

from alphascout.features.cointegration import _eg_chunk, select_pairs
from alphascout.features.core import momentum, roll_mean, roll_std, volume_imbalance, zscore
from alphascout.features.hurst import hurst_exponent, rolling_hurst
from alphascout.features.leakage import truncation_audit


def test_rolling_stats_match_numpy():
    x = np.random.default_rng(0).normal(size=(200, 3))
    assert np.allclose(roll_mean(x, 20)[50], x[31:51].mean(0))
    assert np.allclose(roll_std(x, 20)[50], x[31:51].std(0, ddof=1))
    assert np.isnan(roll_mean(x, 20)[18]).all()


def test_hurst_separates_regimes():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.normal(size=(15, 2000)), axis=1)
    ou = np.zeros((15, 2000))
    for t in range(1, 2000):
        ou[:, t] = 0.9 * ou[:, t - 1] + rng.normal(size=15)
    tr = np.cumsum(np.cumsum(rng.normal(size=(15, 2000)), axis=1) * 0.2 + 0.5, axis=1)
    h_rw, h_ou, h_tr = (np.mean([hurst_exponent(x) for x in a]) for a in (rw, ou, tr))
    assert abs(h_rw - 0.5) < 0.05 and h_ou < 0.42 and h_tr > 0.7


def test_engle_granger_matches_statsmodels(panel):
    X = panel["logp"][300:552]
    idx = np.flatnonzero(np.isfinite(X).all(0))[:10]
    I, J = np.triu_indices(len(idx), 1)
    _, t, *_ = _eg_chunk(X[:, idx], I, J)
    sm = np.array([coint(X[:, idx[i]], X[:, idx[j]], trend="c", maxlag=1, autolag=None)[0] for i, j in zip(I, J)])
    assert np.allclose(t, sm, atol=1e-8)


def test_pair_selection_uses_only_window_and_finds_planted_pairs():
    rng = np.random.default_rng(1)
    F, A = 252, 30
    X = np.cumsum(rng.normal(0, 0.015, (F, A)), axis=0)
    s = np.zeros(F)
    for t in range(1, F):
        s[t] = 0.85 * s[t - 1] + rng.normal(0, 0.01)
    X[:, 7] = 1.0 * X[:, 3] + s     # planted cointegrated pair (3, 7)
    pairs = select_pairs(X, k_max=3, p_max=0.01)
    assert any({p.i, p.j} == {3, 7} for p in pairs)


def test_leakage_detector_catches_known_leaks_and_passes_causal(panel):
    ok = truncation_audit(lambda p: momentum(p["logp"], 60, 5), panel, "mom")
    shift = truncation_audit(lambda p: np.vstack([p["ret"][1:], np.full((1, p.A), np.nan)]), panel, "fwd ret")
    gz = truncation_audit(lambda p: (p["logp"] - np.nanmean(p["logp"], 0)) / np.nanstd(p["logp"], 0), panel, "global z")
    centred = truncation_audit(lambda p: np.where(np.isfinite(p["ret"]), p["ret"] - np.nanmean(p["ret"]), np.nan), panel, "demean")
    assert not ok.leaked and shift.leaked and gz.leaked and centred.leaked
    assert not truncation_audit(lambda p: rolling_hurst(p["logp"], 120, 5), panel, "hurst").leaked
    assert not truncation_audit(lambda p: zscore(p["logp"], 20), panel, "z").leaked
    assert not truncation_audit(lambda p: volume_imbalance(p, 10), panel, "vi").leaked
