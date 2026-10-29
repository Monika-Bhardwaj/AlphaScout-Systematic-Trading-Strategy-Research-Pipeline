"""Can the validation framework catch its own weaknesses?

(1) iid-noise experiment: N strategies that are pure random noise (optionally fat-tailed). Measures
    - mean PBO (should be ~0.5 for skill-less strategies: the IS winner is a coin flip OOS),
    - how often the cherry-picked best strategy looks "significant" WITHOUT correction (high),
    - how often the corrected gate (DSR and PBO) lets anything through (should be <= alpha),
    and, with one planted edge of known Sharpe, the gate's power (it must not be blind either).
(2) pipeline-level experiment: the full pipeline (real generators, costs, leakage audit) on a market
    with no tradable structure; the family-wise false-positive rate of the final gate.
"""
from __future__ import annotations

import numpy as np

from ..data.synthetic import noise_world
from ..pipeline import PipelineConfig, run_pipeline
from ..validation.dsr import deflated_sharpe
from ..validation.pbo import cscv_pbo
from ..validation.reality_check import romano_wolf


def iid_noise_experiment(n_strategies: int = 200, T: int = 1500, repeats: int = 60, seed: int = 0,
                         n_slices: int = 16, alpha: float = 0.05, pbo_max: float = 0.40,
                         planted_ann_sharpes=(0.0, 1.0, 2.0, 3.0), dist: str = "normal", ppy: int = 252,
                         rw_boot: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for s in planted_ann_sharpes:
        pbos, naive, gate, planted_hit, false_hits = [], 0, 0, 0, 0
        for _ in range(repeats):
            R = rng.standard_t(4, (T, n_strategies)) / np.sqrt(2.0) if dist == "t4" else rng.normal(size=(T, n_strategies))
            R = R * 0.01
            if s > 0:
                R[:, 0] += 0.01 * s / np.sqrt(ppy)
            p, d = cscv_pbo(R, n_slices), deflated_sharpe(R, sr_var_method="sampling")
            rw = romano_wolf(R, rw_boot, 20.0, int(rng.integers(1 << 30)))
            best = int(np.nanargmax(d.sr))
            hits = (d.dsr >= 1 - alpha) & (rw <= alpha)      # same per-strategy gate the pipeline uses
            naive += int(d.psr0[best] >= 1 - alpha)
            pbos.append(p.pbo)
            gate += int(hits.any() and p.pbo <= pbo_max)
            planted_hit += int(s > 0 and hits[0] and p.pbo <= pbo_max)
            false_hits += int(hits[1:].sum() if s > 0 else hits.sum())
        out[f"planted_sharpe={s:g}"] = {
            "mean_pbo": float(np.mean(pbos)), "sd_pbo": float(np.std(pbos)),
            "naive_best_looks_significant": naive / repeats,
            "corrected_gate_passes_anything": gate / repeats,
            "planted_detected": planted_hit / repeats if s > 0 else None,
            "mean_false_dsr_hits_per_run": false_hits / repeats,
        }
    return {"n_strategies": n_strategies, "T": T, "repeats": repeats, "dist": dist, "alpha": alpha,
            "pbo_max": pbo_max, "results": out}


def pipeline_noise_experiment(seeds=range(5), n_assets: int = 60, n_days: int = 1500,
                              cfg: PipelineConfig | None = None) -> dict:
    cfg = cfg or PipelineConfig()
    rows = []
    for sd in seeds:
        res = run_pipeline(noise_world(n_assets, n_days, sd), cfg, verbose=False)
        g = res["gate"]
        rows.append({"seed": sd, "pbo": g["pbo"], "best_raw_sharpe": g["best_raw_sharpe_annualised"],
                     "sr0": g["sr0_annualised"], "naive_significant": g["n_naive_significant"],
                     "dsr_significant": g["n_dsr_significant"], "survivors": g["n_survivors"]})
    n = len(rows)
    return {"runs": rows,
            "naive_any_significant_rate": sum(r["naive_significant"] > 0 for r in rows) / n,
            "final_gate_false_positive_rate": sum(r["survivors"] > 0 for r in rows) / n,
            "mean_pbo": float(np.mean([r["pbo"] for r in rows]))}
