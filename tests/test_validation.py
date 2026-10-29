import numpy as np

from alphascout.robustness.noise_test import iid_noise_experiment
from alphascout.validation.dsr import deflated_sharpe, expected_max_sharpe
from alphascout.validation.pbo import cscv_pbo
from alphascout.validation.reality_check import romano_wolf, stationary_bootstrap_indices
from alphascout.validation.walkforward import PurgedKFold, purged_cv_select, walk_forward_select


def test_pbo_is_about_half_on_pure_noise_and_low_with_real_edge():
    rng = np.random.default_rng(0)
    noise = [cscv_pbo(rng.normal(size=(640, 100)), 8).pbo for _ in range(80)]
    assert 0.40 < np.mean(noise) < 0.60
    R = rng.normal(size=(640, 100)) * 0.01
    R[:, 0] += 0.01 * 0.4      # per-period Sharpe 0.4: unmistakable edge
    assert cscv_pbo(R, 8).pbo < 0.15


def test_deflated_sharpe_controls_false_positives_that_naive_testing_does_not():
    r = iid_noise_experiment(n_strategies=150, T=1000, repeats=40, seed=1, planted_ann_sharpes=(0.0,), rw_boot=150)
    res = r["results"]["planted_sharpe=0"]
    assert res["naive_best_looks_significant"] > 0.8        # cherry-picking "works" ~always on noise
    assert res["corrected_gate_passes_anything"] <= 0.10    # the corrected gate does not


def test_gate_has_power_for_strong_edge_and_survives_fat_tails():
    r = iid_noise_experiment(n_strategies=100, T=1500, repeats=25, seed=2, planted_ann_sharpes=(0.0, 3.5), dist="t4", rw_boot=150)
    assert r["results"]["planted_sharpe=0"]["corrected_gate_passes_anything"] <= 0.12
    assert r["results"]["planted_sharpe=3.5"]["planted_detected"] >= 0.8


def test_expected_max_sharpe_grows_with_trials():
    v = 1.0 / 1000
    assert expected_max_sharpe(1, v) == 0 < expected_max_sharpe(10, v) < expected_max_sharpe(1000, v)


def test_psr_dsr_handle_degenerate_columns():
    R = np.random.default_rng(0).normal(size=(500, 20)) * 0.01
    R[:, 3] = 0.0
    d = deflated_sharpe(R, n_trials=25)
    assert np.isnan(d.dsr[3]) and np.isfinite(d.dsr[0]) and d.n_trials == 25


def test_purged_kfold_has_no_label_overlap_and_respects_embargo():
    n, h, emb = 500, 21, 5
    t1 = np.minimum(np.arange(n) + h, n - 1)
    for train, test in PurgedKFold(5, emb).split(t1):
        span = (test[0], t1[test].max())
        assert not np.any((train <= span[1]) & (t1[train] >= span[0]))       # no train label touches test span
        assert not np.any((train > test[-1]) & (train <= test[-1] + emb))    # embargo respected
        assert len(train) > 0.5 * n * 0.7


def test_walk_forward_never_uses_test_rows_to_pick():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(1200, 30)) * 0.01
    w = walk_forward_select(R, train_len=400, test_len=100, gap=5)
    for start, k, _ in w.picks:
        assert start >= 405
        assert np.isfinite(w.oos[start])
    assert np.isnan(w.oos[:405]).all()
    # a strategy that is great only AFTER the first test window cannot be picked for that window
    R2 = R.copy(); R2[600:, 0] += 0.02
    w2 = walk_forward_select(R2, train_len=400, test_len=100, gap=5)
    assert all(k != 0 for s, k, _ in w2.picks if s + 100 <= 600)
    assert purged_cv_select(R, 21, 5, 5).oos.shape == (1200,)


def test_stationary_bootstrap_produces_valid_blocks():
    idx = stationary_bootstrap_indices(400, 100, 20.0, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 400
    break_rate = np.mean(np.diff(idx, axis=1) % 400 != 1)
    assert 0.03 < break_rate < 0.08          # ~1/mean_block


def test_romano_wolf_controls_fwer_on_noise_and_detects_real_edge():
    rng = np.random.default_rng(3)
    fwer = np.mean([(romano_wolf(rng.normal(size=(800, 60)), 200, 20, r) <= 0.05).any() for r in range(40)])
    assert fwer <= 0.15                       # nominal 0.05; loose bound for 40 runs
    R = rng.normal(size=(800, 60)) * 0.01
    R[:, 0] += 0.01 * 0.2
    p = romano_wolf(R, 300, 20, 1)
    assert p[0] <= 0.05 and np.nanmin(p[1:]) > 0.05


def test_cross_section_null_variance_is_underpowered_on_heterogeneous_candidates_sampling_null_is_not():
    """Half the candidates are genuinely bad (net-of-cost losers), a few genuinely good: the
    cross-sectional Sharpe variance measures that skill dispersion, not luck, and inflates SR0."""
    rng = np.random.default_rng(5)
    T, N = 1500, 100
    R = rng.normal(size=(T, N)) * 0.01
    R[:, :50] -= 0.01 * 0.07            # per-period Sharpe -0.07 (about -1.1 annualised)
    R[:, 50:54] += 0.01 * 0.12          # per-period Sharpe +0.12 (about +1.9 annualised)
    cs, sm = deflated_sharpe(R, sr_var_method="cross_section"), deflated_sharpe(R, sr_var_method="sampling")
    assert cs.sr0 > 2 * sm.sr0
    assert (sm.dsr[50:54] > 0.95).sum() >= 3 and (cs.dsr[50:54] > 0.95).sum() < (sm.dsr[50:54] > 0.95).sum()
    assert (sm.dsr[:50] > 0.95).sum() == 0 and (sm.dsr[54:] > 0.95).sum() <= 1
