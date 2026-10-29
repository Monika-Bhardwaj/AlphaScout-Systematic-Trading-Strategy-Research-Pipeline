import dataclasses

import numpy as np
import pytest

import alphascout.pipeline as pl_mod
from alphascout.data.synthetic import noise_world
from alphascout.pipeline import PipelineConfig, run_pipeline


@dataclasses.dataclass(frozen=True)
class Leaky:
    family: str = "leaky"
    name_: str = "leaky_peek"

    @property
    def name(self):
        return self.name_

    def weights(self, panel, cache):
        r = np.nan_to_num(panel["ret"])
        W = np.zeros_like(r)
        W[:-1] = np.sign(r[1:]) / panel.A     # peeks at tomorrow
        return W


def test_pipeline_refuses_to_run_with_a_leaky_strategy(monkeypatch):
    from alphascout.strategies.generator import generate_candidates
    monkeypatch.setattr(pl_mod, "generate_candidates", lambda n, seed: [Leaky()] + generate_candidates(4, 0))
    with pytest.raises(RuntimeError, match="Leakage audit failed"):
        run_pipeline(noise_world(40, 700, 0), PipelineConfig(), verbose=False)


def test_pipeline_end_to_end_reports_everything_on_noise():
    res = run_pipeline(noise_world(40, 900, 1), PipelineConfig(n_candidates=24, wf_train=252), verbose=False)
    assert res["gate"]["n_survivors"] == 0 and res["finalists_are_diagnostic_only"]
    assert all("cost_stress" in f and "regimes_research" in f for f in res["finalists"])
    assert set(res["finalists"][0]["cost_stress"]) == {"1x", "2x", "3x"}
    assert res["timing"]["accounting_speedup_x"] > 5 and res["timing"]["accounting_equivalence_max_abs_diff"] < 1e-10
    assert res["config"]["n_candidates"] == 24


def test_pipeline_finds_the_planted_edge_and_only_that(tmp_path):
    from alphascout.data.synthetic import generate_synthetic
    raw = generate_synthetic(80, 1800, seed=5, planted_momentum=3.0)
    res = run_pipeline(raw, PipelineConfig(n_candidates=36, rw_bootstrap=300, wf_train=252), outdir=str(tmp_path), verbose=False)
    surv = [f for f in res["finalists"] if f["survivor"]]
    assert surv and not res["finalists_are_diagnostic_only"]
    assert all(f["family"] == "momentum" for f in surv)               # the only structure that exists
    ho = np.array([f["holdout_sharpe"] for f in surv])                 # a 360-day hold-out has SE ~0.8 on a Sharpe,
    rs = np.array([f["research_sharpe"] for f in surv])                # so test the distribution, not each point:
    assert np.median(ho) > 1.0 and (ho > 0).mean() >= 0.8              # the edge persists on untouched data ...
    assert ho.mean() < rs.mean()                                       # ... but shrinks (winner's curse), as it should
    assert all(f["dsr"] >= 0.95 and f["rw_adjusted_p"] <= 0.05 for f in surv)
    assert (tmp_path / "summary.json").exists() and (tmp_path / "candidates.csv").exists()
    sheets = list(tmp_path.glob("tearsheet_*.html"))
    assert sheets and "<img" in sheets[0].read_text()
