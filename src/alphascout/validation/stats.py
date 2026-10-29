from __future__ import annotations

import numpy as np


def sharpe_cols(R: np.ndarray) -> np.ndarray:
    """Per-period (non-annualised) Sharpe per column; zero-variance columns -> NaN."""
    mu = np.nanmean(R, axis=0)
    sd = np.nanstd(R, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 1e-14, mu / sd, np.nan)
