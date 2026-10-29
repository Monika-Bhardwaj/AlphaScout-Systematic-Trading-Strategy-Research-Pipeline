"""AlphaScout pipeline, milestone 1: PIT data -> leakage audit -> vectorised screen of N candidates
-> multiple-testing gate (PBO + deflated Sharpe) -> walk-forward / purged-CV evidence -> cost stress,
regimes, risk and untouched hold-out for the finalists."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .backtest.costs import CostModel
from .backtest.engine import pnl_reference_loop, pnl_vectorized
from .backtest.event_sim import ExecConfig, simulate
from .data.pit_store import PITStore
from .data.raw import RawMarketData
from .features.cache import FeatureCache
from .features.core import momentum, volume_imbalance, vwap_deviation, zscore
from .features.hurst import rolling_hurst
from .features.leakage import truncation_audit
from .risk.metrics import sharpe, summary
from .risk.regimes import market_regimes, regime_breakdown
from .risk.sizing import vol_target
from .strategies.generator import generate_candidates
from .validation.dsr import deflated_sharpe
from .validation.pbo import cscv_pbo
from .validation.reality_check import romano_wolf
from .validation.walkforward import purged_cv_select, walk_forward_select


@dataclass
class PipelineConfig:
    n_candidates: int | None = None          # None -> default grid
    research_frac: float = 0.8               # last (1 - frac) of dates is a never-touched hold-out
    n_slices: int = 16
    dsr_alpha: float = 0.05
    pbo_max: float = 0.40
    rw_bootstrap: int = 1000
    rw_block: float = 20.0
    cost: CostModel = field(default_factory=CostModel)
    wf_train: int = 504
    wf_test: int = 63
    wf_gap: int = 5
    cv_splits: int = 5
    cv_embargo: int = 5
    cv_horizon: int = 21
    vol_target: float = 0.10
    stress_multipliers: tuple = (1.0, 2.0, 3.0)
    audit_per_family: int = 1
    n_finalists_if_none: int = 5
    seed: int = 0


def _j(o):
    if isinstance(o, dict):
        return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def leakage_audit(panel, cands, per_family: int = 1) -> list:
    reps = [
        truncation_audit(lambda p: momentum(p["logp"], 60, 5), panel, "feature:momentum"),
        truncation_audit(lambda p: zscore(p["logp"], 20), panel, "feature:zscore"),
        truncation_audit(lambda p: rolling_hurst(p["logp"], 120, 5), panel, "feature:hurst"),
        truncation_audit(lambda p: volume_imbalance(p, 10), panel, "feature:volume_imbalance"),
        truncation_audit(lambda p: vwap_deviation(p), panel, "feature:vwap_deviation"),
    ]
    seen: dict = {}
    for c in cands:
        seen.setdefault(c.family, [])
        if len(seen[c.family]) < per_family:
            seen[c.family].append(c)
    for fam, cs in seen.items():
        for c in cs:
            reps.append(truncation_audit(lambda p, c=c: c.weights(p, FeatureCache(p)), panel, f"strategy:{c.name}"))
    return reps


def run_pipeline(raw: RawMarketData, cfg: PipelineConfig | None = None, outdir: str | None = None,
                 verbose: bool = True) -> dict:
    cfg = cfg or PipelineConfig()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    timing: dict = {}
    ppy = raw.periods_per_year
    cost = CostModel(one_way_bps=raw.default_cost_bps, borrow_bps_annual=cfg.cost.borrow_bps_annual,
                     multiplier=cfg.cost.multiplier) if cfg.cost == CostModel() else cfg.cost

    t0 = time.perf_counter()
    store = PITStore(raw)
    panel = store.build_panel()
    timing["data_build_s"] = time.perf_counter() - t0
    rej = store.rejection_log()
    log(f"[data] {panel.T} dates x {panel.A} symbols (delisted incl.), {rej.height} rejection-log rows")

    cands = generate_candidates(cfg.n_candidates, cfg.seed)
    N = len(cands)

    t0 = time.perf_counter()
    reports = leakage_audit(panel, cands, cfg.audit_per_family)
    timing["leakage_audit_s"] = time.perf_counter() - t0
    for r in reports:
        log(str(r))
    if any(r.leaked for r in reports):
        raise RuntimeError("Leakage audit failed; refusing to run the pipeline:\n" + "\n".join(map(str, reports)))

    # ---- stage 1: vectorised screen over all N ------------------------------------------------
    T = panel.T
    n_res = int(T * cfg.research_frac)
    cache = FeatureCache(panel)
    R = np.zeros((T, N)); turn = np.zeros(N); gross_exp = np.zeros(N)
    t_w = t_p = 0.0
    t0 = time.perf_counter()
    for k, c in enumerate(cands):
        a = time.perf_counter()
        W = c.weights(panel, cache)
        b = time.perf_counter()
        net, _, tn = pnl_vectorized(W, panel["ret"], cost, ppy)
        t_p += time.perf_counter() - b
        t_w += b - a
        R[:, k] = net
        turn[k] = tn[:n_res].mean()
        gross_exp[k] = np.abs(W[:n_res]).sum(axis=1).mean()
    timing["screen_total_s"] = time.perf_counter() - t0
    timing["screen_weight_generation_s"] = t_w
    timing["screen_pnl_accounting_s"] = t_p
    log(f"[screen] {N} candidates in {timing['screen_total_s']:.1f}s "
        f"(weights {t_w:.1f}s, vectorised PnL accounting {t_p:.2f}s)")

    # speedup of the vectorised accounting vs the naive per-day/per-asset loop (measured, not guessed)
    Wm = cands[0].weights(panel, cache)
    ref_cost = CostModel(cost.one_way_bps, cost.borrow_bps_annual, cost.multiplier)
    a = time.perf_counter(); v, _, _ = pnl_vectorized(Wm, panel["ret"], ref_cost, ppy); tv = time.perf_counter() - a
    a = time.perf_counter(); l, _, _ = pnl_reference_loop(Wm, panel["ret"], ref_cost, ppy); tl = time.perf_counter() - a
    timing["accounting_vectorised_per_candidate_s"] = tv
    timing["accounting_loop_per_candidate_s"] = tl
    timing["accounting_speedup_x"] = tl / max(tv, 1e-12)
    timing["accounting_equivalence_max_abs_diff"] = float(np.abs(v - l).max())

    # ---- stage 2: the multiple-testing gate on the research window ----------------------------
    Rr = R[:n_res]
    dsr = deflated_sharpe(Rr, n_trials=N, sr_var_method="sampling")
    dsr_cs = deflated_sharpe(Rr, n_trials=N, sr_var_method="cross_section")
    rw_p = romano_wolf(Rr, cfg.rw_bootstrap, cfg.rw_block, cfg.seed)
    pbo = cscv_pbo(Rr, cfg.n_slices)
    ann = np.sqrt(ppy)
    is_survivor = ((dsr.dsr >= 1 - cfg.dsr_alpha) & np.isfinite(dsr.dsr)
                   & (rw_p <= cfg.dsr_alpha) & (pbo.pbo <= cfg.pbo_max))
    log(f"[gate] N={N} trials | PBO={pbo.pbo:.3f} (gate <= {cfg.pbo_max}) | best raw Sharpe={np.nanmax(dsr.sr) * ann:.2f} ann | "
        f"SR0 (expected max from luck) sampling-null={dsr.sr0 * ann:.2f}, cross-section-null={dsr_cs.sr0 * ann:.2f} ann")
    log(f"[gate] significant: naive(uncorrected 5%)={int((dsr.p_naive < 0.05).sum())}, BH-FDR={int((dsr.q_bh < 0.05).sum())}, "
        f"DSR(sampling)={int((dsr.dsr >= 1 - cfg.dsr_alpha).sum())}, DSR(cross-section)={int((dsr_cs.dsr >= 1 - cfg.dsr_alpha).sum())}, "
        f"Romano-Wolf FWER={int((rw_p <= cfg.dsr_alpha).sum())} | SURVIVORS (DSR & RW & PBO)={int(is_survivor.sum())}")

    # ---- stage 3: walk-forward and purged CV evidence about the *selection procedure* ---------
    wf = walk_forward_select(Rr, cfg.wf_train, cfg.wf_test, cfg.wf_gap)
    cv = purged_cv_select(Rr, cfg.cv_horizon, cfg.cv_splits, cfg.cv_embargo)
    wf_sr = sharpe(wf.oos[np.isfinite(wf.oos)], ppy) if np.isfinite(wf.oos).any() else None
    cv_sr = sharpe(cv.oos[np.isfinite(cv.oos)], ppy)
    log(f"[oos] walk-forward selection OOS Sharpe={wf_sr}, purged-CV selection OOS Sharpe={cv_sr:.2f}")

    # ---- stage 4: finalists ---------------------------------------------------------------------
    order = np.argsort(-np.nan_to_num(dsr.sr, nan=-9))
    fin_idx = [int(k) for k in order if is_survivor[k]][:10]
    diagnostic_only = False
    if not fin_idx:
        fin_idx = [int(k) for k in order[: cfg.n_finalists_if_none]]
        diagnostic_only = True
        log("[gate] NOTHING SURVIVES. Reporting top candidates by raw Sharpe as diagnostics only "
            "(they failed the corrected-significance gate).")
    reg = market_regimes(panel["ret"], panel.listed)
    finalists = []
    series: dict = {}
    t_event = 0.0
    n_event_runs = 0
    for k in fin_idx:
        c = cands[k]
        W = c.weights(panel, cache)
        stress = {}
        for m in cfg.stress_multipliers:
            net, _, _ = pnl_vectorized(W, panel["ret"], cost.scaled(m), ppy)
            stress[f"{m:g}x"] = {"research_sharpe": sharpe(net[:n_res], ppy), "holdout_sharpe": sharpe(net[n_res:], ppy),
                                 "research_ann_return": float(np.nanmean(net[:n_res]) * ppy)}
        net, grs, tn = pnl_vectorized(W, panel["ret"], cost, ppy)
        vt, lev = vol_target(net, np.abs(W).sum(axis=1), cfg.vol_target, ppy=ppy, rescale_cost_bps=cost.one_way_bps)
        # paper-trading replay on the untouched hold-out: event-driven fills, partial fills, impact, latency
        base = ExecConfig(fees_bps=max(cost.one_way_bps - 3.0, 0.0), borrow_bps_annual=cost.borrow_bps_annual)
        scen = {"base (fill next bar VWAP, 1x costs)": base,
                "2x costs": ExecConfig(**{**base.__dict__, "cost_multiplier": 2.0}),
                "3x costs": ExecConfig(**{**base.__dict__, "cost_multiplier": 3.0}),
                "latency +1 bar": ExecConfig(**{**base.__dict__, "delay_bars": 2}),
                "latency +2 bars": ExecConfig(**{**base.__dict__, "delay_bars": 3})}
        replay = {}
        for label, ec in scen.items():
            a = time.perf_counter()
            sr = simulate(panel, W, ec, start=n_res, ppy=ppy)
            t_event += time.perf_counter() - a
            n_event_runs += 1
            replay[label] = {"sharpe": sharpe(sr.returns, ppy), "total_return": float(sr.equity[-1] / ec.capital - 1.0),
                             "fill_ratio": sr.fill_ratio, "cost_fraction_of_capital": sr.cost_total / ec.capital,
                             "n_events": len(sr.events), "events": sr.events[:8], "halted_bars": sr.halted_bars}
        series[c.name] = (net, vt)
        finalists.append({
            "paper_replay_holdout": replay,
            "name": c.name, "family": c.family, "survivor": bool(is_survivor[k]),
            "research_sharpe": float(dsr.sr[k] * ann), "dsr": float(dsr.dsr[k]), "p_naive": float(dsr.p_naive[k]),
            "dsr_cross_section": float(dsr_cs.dsr[k]), "rw_adjusted_p": float(rw_p[k]),
            "q_bh": float(dsr.q_bh[k]), "avg_daily_turnover": float(turn[k]), "avg_gross": float(gross_exp[k]),
            "holdout_sharpe": sharpe(net[n_res:], ppy),
            "cost_stress": stress,
            "regimes_research": regime_breakdown(net[:n_res], {kk: v[:n_res] for kk, v in reg.items()}, ppy),
            "risk_research_unscaled": summary(net[:n_res], ppy),
            "risk_research_vol_targeted": summary(vt[:n_res], ppy),
            "risk_holdout_vol_targeted": summary(vt[n_res:], ppy),
            "mean_leverage_voltarget": float(np.mean(lev[lev > 0])) if (lev > 0).any() else None,
        })

    timing["event_sim_per_run_s"] = t_event / max(n_event_runs, 1)
    timing["event_sim_runs"] = n_event_runs
    timing["event_vs_vectorised_accounting_x"] = timing["event_sim_per_run_s"] / max(timing["accounting_vectorised_per_candidate_s"], 1e-12)
    fam = np.array([c.family for c in cands])
    fam_stats = {f: {"n": int((fam == f).sum()), "median_research_sharpe": float(np.nanmedian(dsr.sr[fam == f]) * ann),
                     "best_research_sharpe": float(np.nanmax(dsr.sr[fam == f]) * ann) if np.isfinite(dsr.sr[fam == f]).any() else None,
                     "survivors": int(is_survivor[fam == f].sum())} for f in sorted(set(fam))}
    result = {
        "config": {"n_candidates": N, "research_rows": n_res, "holdout_rows": T - n_res, "periods_per_year": ppy,
                   "cost_one_way_bps": cost.one_way_bps, "dsr_alpha": cfg.dsr_alpha, "pbo_max": cfg.pbo_max,
                   "n_slices": cfg.n_slices, "data": raw.name},
        "data_quality": {"rejection_log_rows": int(rej.height),
                         "by_rule": {r: int(n) for r, n in rej.group_by("rule").len().iter_rows()} if rej.height else {}},
        "leakage_audit": [str(r) for r in reports],
        "gate": {"pbo": pbo.pbo, "pbo_prob_oos_loss": pbo.prob_oos_loss, "pbo_degradation_slope": pbo.degradation_slope,
                 "sr0_annualised": dsr.sr0 * ann, "sr0_cross_section_annualised": dsr_cs.sr0 * ann,
                 "sr_var_per_period_sampling": dsr.sr_var, "sr_var_per_period_cross_section": dsr_cs.sr_var,
                 "n_dsr_cross_section_significant": int((dsr_cs.dsr >= 1 - cfg.dsr_alpha).sum()),
                 "n_romano_wolf_significant": int((rw_p <= cfg.dsr_alpha).sum()),
                 "best_raw_sharpe_annualised": float(np.nanmax(dsr.sr) * ann),
                 "n_naive_significant": int((dsr.p_naive < 0.05).sum()), "n_dsr_significant": int((dsr.dsr >= 0.95).sum()),
                 "n_bh_fdr_significant": int((dsr.q_bh < 0.05).sum()), "n_survivors": int(is_survivor.sum())},
        "oos_selection": {"walk_forward_sharpe": wf_sr, "purged_cv_sharpe": cv_sr,
                          "walk_forward_windows": len(wf.picks), "purged_cv_folds": len(cv.picks)},
        "families": fam_stats, "finalists_are_diagnostic_only": diagnostic_only, "finalists": finalists,
        "timing": timing,
    }
    result = _j(result)
    if outdir:
        out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        pl.DataFrame({"name": [c.name for c in cands], "family": fam.tolist(), "research_sharpe_ann": (dsr.sr * ann).tolist(),
                      "dsr": dsr.dsr.tolist(), "dsr_cross_section": dsr_cs.dsr.tolist(), "rw_adjusted_p": rw_p.tolist(), "p_naive": dsr.p_naive.tolist(), "q_bh": dsr.q_bh.tolist(),
                      "avg_daily_turnover": turn.tolist(), "survivor": is_survivor.tolist()}).write_csv(out / "candidates.csv")
        if rej.height:
            rej.write_csv(out / "rejection_log.csv")
        np.save(out / "returns_matrix.npy", R)
        try:
            from .report.tearsheet import make_tearsheet
            for f in result["finalists"]:
                net, vt = series[f["name"]]
                html_ = make_tearsheet(f, panel.dates, net, vt, n_res, f["paper_replay_holdout"], result["gate"] | {"n_trials": N}, ppy)
                (out / f"tearsheet_{f['name']}.html").write_text(html_)
        except ImportError:
            log("[report] matplotlib not installed; skipping tearsheets")
    result["_series"] = series
    return result
