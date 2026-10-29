"""Walk-forward selection and purged/embargoed K-fold selection on a (T, N) strategy-return matrix.

Both answer: "if I had to pick the best candidate using only the allowed training rows, how does
that pick do on rows it never saw?" Neither replaces the multiple-testing correction; they are
extra out-of-sample evidence about the selection procedure."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import sharpe_cols


@dataclass
class WFResult:
    oos: np.ndarray                 # (T,) stitched OOS returns, NaN where never tested
    picks: list                     # (test_start, strategy_idx, train_sharpe_per_period)
    window_sharpes: list            # per-window OOS per-period Sharpe


def _argbest(R_train: np.ndarray) -> tuple[int, float]:
    sr = sharpe_cols(R_train)
    if not np.isfinite(sr).any():
        return -1, float("nan")
    k = int(np.nanargmax(sr))
    return k, float(sr[k])


def walk_forward_select(R: np.ndarray, train_len: int = 504, test_len: int = 63, gap: int = 5,
                        anchored: bool = False) -> WFResult:
    T, N = R.shape
    oos = np.full(T, np.nan)
    picks, wsr = [], []
    start = train_len + gap
    while start < T:
        end = min(start + test_len, T)
        lo = 0 if anchored else start - gap - train_len
        k, sr_tr = _argbest(R[lo: start - gap])
        if k >= 0:
            oos[start:end] = R[start:end, k]
            picks.append((start, k, sr_tr))
            x = R[start:end, k]
            wsr.append(float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else float("nan"))
        start = end
    return WFResult(oos, picks, wsr)


class PurgedKFold:
    """K contiguous folds. For test fold [a, b]: a train observation i with label interval
    [i, t1[i]] is purged if that interval overlaps the test span [a, max t1 over the test fold];
    rows in (b, b + embargo] are also dropped (serial-correlation leakage after the test block)."""

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        self.n_splits, self.embargo = n_splits, embargo

    def split(self, t1: np.ndarray):
        n = len(t1)
        idx = np.arange(n)
        for te in np.array_split(idx, self.n_splits):
            a, b = te[0], te[-1]
            span_end = t1[te].max()
            overlap = (idx <= span_end) & (t1 >= a)
            emb = (idx > b) & (idx <= b + self.embargo)
            yield idx[~overlap & ~emb], te


def purged_cv_select(R: np.ndarray, horizon: int = 21, n_splits: int = 5, embargo: int = 5) -> WFResult:
    T, N = R.shape
    t1 = np.minimum(np.arange(T) + horizon, T - 1)
    oos = np.full(T, np.nan)
    picks, wsr = [], []
    for train, test in PurgedKFold(n_splits, embargo).split(t1):
        k, sr_tr = _argbest(R[train])
        if k < 0:
            continue
        oos[test] = R[test, k]
        picks.append((int(test[0]), k, sr_tr))
        x = R[test, k]
        wsr.append(float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else float("nan"))
    return WFResult(oos, picks, wsr)
