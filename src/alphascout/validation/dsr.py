"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

The deflation: the best of N tried strategies has an inflated Sharpe even if every strategy is
noise. SR0 is the expected maximum Sharpe among N skill-less trials; DSR is the probability that a
strategy's true Sharpe exceeds SR0 given its sample length, skew and kurtosis. All Sharpe ratios
here are per-period; the formulas are unit-consistent in per-period terms."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.stats import norm

from .stats import sharpe_cols

EULER = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    if n_trials < 2 or not np.isfinite(sr_var) or sr_var <= 0:
        return 0.0
    return float(np.sqrt(sr_var) * ((1 - EULER) * norm.ppf(1 - 1.0 / n_trials)
                                    + EULER * norm.ppf(1 - 1.0 / (n_trials * np.e))))


def psr(sr, sr_star, n_obs, skew, kurt):
    """P(true SR > sr_star). kurt is raw (normal = 3)."""
    d2 = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    return norm.cdf((sr - sr_star) * np.sqrt(n_obs - 1) / np.sqrt(np.maximum(d2, 1e-12)))


@dataclass
class DSRResult:
    sr: np.ndarray          # per-period Sharpe of each strategy
    sr0: float              # expected max Sharpe under the null of N skill-less trials
    sr_var: float           # variance of Sharpe estimates across trials
    dsr: np.ndarray         # P(true SR > SR0)
    psr0: np.ndarray        # naive: P(true SR > 0), no multiple-testing correction
    p_naive: np.ndarray     # 1 - psr0
    q_bh: np.ndarray        # Benjamini-Hochberg adjusted p across all N (cross-check)
    n_trials: int
    n_obs: int


def deflated_sharpe(R: np.ndarray, n_trials: int | None = None, sr_var_method: str = "cross_section") -> DSRResult:
    """R: (T, N) net returns of ALL tried strategies (not just the winner).
    n_trials >= N lets degenerate (never-traded) candidates count as trials.

    sr_var_method sets V, the variance of Sharpe estimates among skill-less trials, which scales SR0:
      "cross_section": variance of the N observed Sharpes (the paper's practical estimator). Faithful, but
          when the candidate set is heterogeneous (some families genuinely good, some genuinely bad after
          costs) it measures real dispersion of skill, inflates SR0 and kills power (seen in the planted
          demo: SR0 2.98 vs best raw Sharpe 2.40 despite a confirmed hold-out edge).
      "sampling": the null sampling variance of a Sharpe estimate, 1/(T-1) (the skew/kurtosis terms vanish
          at SR=0). Independent of how good or bad the candidates are. Assumes roughly uncorrelated
          returns, so the pipeline pairs it with a block-bootstrap Reality Check that does not."""
    T, N = R.shape
    n_trials = max(int(n_trials or N), N)
    sr = sharpe_cols(R)
    fin = np.isfinite(sr)
    if sr_var_method == "cross_section":
        sr_var = float(np.var(sr[fin], ddof=1)) if fin.sum() > 2 else float("nan")
    elif sr_var_method == "sampling":
        sr_var = float(np.mean(1.0 / (np.isfinite(R[:, fin]).sum(axis=0) - 1.0))) if fin.any() else float("nan")
    else:
        raise ValueError(sr_var_method)
    sr0 = expected_max_sharpe(n_trials, sr_var)
    sk, ku = np.full(N, np.nan), np.full(N, np.nan)
    for k in np.flatnonzero(fin):
        x = R[np.isfinite(R[:, k]), k]
        sk[k], ku[k] = stats.skew(x), stats.kurtosis(x, fisher=False)
    n_obs = np.isfinite(R).sum(axis=0)
    with np.errstate(invalid="ignore"):
        dsr = psr(sr, sr0, n_obs, sk, ku)
        psr0 = psr(sr, 0.0, n_obs, sk, ku)
    p_naive = 1.0 - psr0
    q = np.full(N, np.nan)
    if fin.sum():
        q[fin] = stats.false_discovery_control(np.clip(p_naive[fin], 0, 1), method="bh")
    return DSRResult(sr, sr0, sr_var, dsr, psr0, p_naive, q, n_trials, int(T))
