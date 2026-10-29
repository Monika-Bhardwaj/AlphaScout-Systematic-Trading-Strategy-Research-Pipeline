#!/usr/bin/env python
"""Add more no-edge markets to the pipeline-level false-positive estimate in docs/noise_results.json."""
import json
import sys
import warnings

warnings.filterwarnings("ignore", message="Sortedness of columns")
from alphascout.pipeline import PipelineConfig
from alphascout.robustness.noise_test import pipeline_noise_experiment

a, b = int(sys.argv[1]), int(sys.argv[2])
path = "docs/noise_results.json"
res = json.load(open(path))
new = pipeline_noise_experiment(range(a, b), 60, 1500, PipelineConfig())
runs = res["pipeline_noise_world"]["runs"] + new["runs"]
n = len(runs)
res["pipeline_noise_world"] = {
    "runs": runs, "n_worlds": n,
    "naive_any_significant_rate": sum(r["naive_significant"] > 0 for r in runs) / n,
    "final_gate_false_positive_rate": sum(r["survivors"] > 0 for r in runs) / n,
    "mean_pbo": sum(r["pbo"] for r in runs) / n}
json.dump(res, open(path, "w"), indent=2)
print({k: v for k, v in res["pipeline_noise_world"].items() if k != "runs"})
