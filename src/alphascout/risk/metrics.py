from __future__ import annotations

import numpy as np
from scipy import stats


def sharpe(r: np.ndarray, ppy: int = 252) -> float:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 2 else 0.0
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")


def sortino(r: np.ndarray, ppy: int = 252) -> float:
    r = r[np.isfinite(r)]
    dd = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if len(r) else 0.0
    return float(r.mean() / dd * np.sqrt(ppy)) if dd > 0 else float("nan")


def max_drawdown(r: np.ndarray) -> float:
    r = np.nan_to_num(r)
    w = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return float((w / peak - 1.0).min())


def cvar(r: np.ndarray, alpha: float = 0.05) -> float:
    """Expected shortfall: mean of the worst alpha fraction of periods (reported as a loss, >0)."""
    r = np.sort(r[np.isfinite(r)])
    k = max(1, int(np.ceil(alpha * len(r))))
    return float(-r[:k].mean())


def summary(r: np.ndarray, ppy: int = 252) -> dict:
    x = r[np.isfinite(r)]
    mdd = max_drawdown(x)
    ann = float(x.mean() * ppy)
    return {
        "n": int(len(x)), "ann_return": ann, "ann_vol": float(x.std(ddof=1) * np.sqrt(ppy)),
        "sharpe": sharpe(x, ppy), "sortino": sortino(x, ppy), "max_drawdown": mdd,
        "cvar_5pct_daily": cvar(x), "calmar": float(ann / abs(mdd)) if mdd < 0 else float("nan"),
        "skew": float(stats.skew(x)), "kurtosis": float(stats.kurtosis(x, fisher=False)),
    }
