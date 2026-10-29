"""Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu, 2015).

Split the T x N return matrix into S time slices. For every way of choosing S/2 slices as
"in-sample" (and the other S/2 as out-of-sample): pick the strategy with the best IS Sharpe and
record its relative rank among all N strategies OOS. PBO = fraction of combinations in which the
IS winner lands in the bottom half OOS (logit of relative rank <= 0). PBO describes the whole
selection procedure over the N candidates, not any single strategy. For skill-less strategies it
should sit near 0.5."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class PBOResult:
    pbo: float
    logits: np.ndarray
    oos_rel_rank: np.ndarray
    is_best_sr: np.ndarray      # IS Sharpe of the selected strategy per combination
    oos_best_sr: np.ndarray     # its OOS Sharpe
    prob_oos_loss: float        # P(selected strategy has OOS Sharpe < 0)
    degradation_slope: float    # slope of OOS vs IS Sharpe of the selected strategy
    n_combos: int
    n_strategies: int
    n_slices: int


def cscv_pbo(R: np.ndarray, n_slices: int = 16) -> PBOResult:
    assert n_slices % 2 == 0 and n_slices >= 4
    R = np.asarray(R, float)
    keep = np.isfinite(R).all(axis=0) & (R.std(axis=0) > 1e-14)   # drop degenerate columns
    R = R[:, keep]
    T, N = R.shape
    S = n_slices
    m = T // S
    R = R[T - m * S:]                                    # drop the oldest remainder rows
    blocks = R.reshape(S, m, N)
    bsum, bsq = blocks.sum(axis=1), (blocks ** 2).sum(axis=1)
    combos = np.array(list(itertools.combinations(range(S), S // 2)))
    C = len(combos)
    sel = np.zeros((C, S))
    sel[np.arange(C)[:, None], combos] = 1.0
    n_is = (S // 2) * m
    tot_s, tot_q = bsum.sum(axis=0), bsq.sum(axis=0)

    def sr(sum_, sq_, n):
        mean = sum_ / n
        var = np.maximum((sq_ - n * mean ** 2) / (n - 1), 1e-18)
        return mean / np.sqrt(var)

    is_sum, is_sq = sel @ bsum, sel @ bsq
    is_sr = sr(is_sum, is_sq, n_is)
    oos_sr = sr(tot_s - is_sum, tot_q - is_sq, n_is)
    best = is_sr.argmax(axis=1)
    rows = np.arange(C)
    oos_best = oos_sr[rows, best]
    rank = (oos_sr < oos_best[:, None]).sum(axis=1) + 1
    omega = rank / (N + 1.0)
    logit = np.log(omega / (1.0 - omega))
    is_best = is_sr[rows, best]
    slope = float(np.polyfit(is_best, oos_best, 1)[0]) if np.std(is_best) > 0 else float("nan")
    return PBOResult(float(np.mean(logit <= 0)), logit, omega, is_best, oos_best,
                     float(np.mean(oos_best < 0)), slope, C, N, S)
