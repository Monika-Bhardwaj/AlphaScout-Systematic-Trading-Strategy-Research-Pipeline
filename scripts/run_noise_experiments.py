#!/usr/bin/env python
"""Prove the validation framework catches its own weaknesses. Writes docs/noise_results.json."""
import json
import sys
import warnings

warnings.filterwarnings("ignore", message="Sortedness of columns")

from alphascout.pipeline import PipelineConfig
from alphascout.robustness.noise_test import iid_noise_experiment, pipeline_noise_experiment

out = {}
print("iid noise (normal)...", flush=True)
out["iid_normal"] = iid_noise_experiment(200, 1500, 60, seed=0)
print("iid noise (student-t4 fat tails)...", flush=True)
out["iid_t4"] = iid_noise_experiment(200, 1500, 60, seed=1, dist="t4")
n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
print(f"full pipeline on {n_seeds} no-edge markets...", flush=True)
out["pipeline_noise_world"] = pipeline_noise_experiment(range(n_seeds), 60, 1500, PipelineConfig())
json.dump(out, open("docs/noise_results.json", "w"), indent=2)
for k in ("iid_normal", "iid_t4"):
    print(k)
    for s, v in out[k]["results"].items():
        print("  ", s, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
p = out["pipeline_noise_world"]
print("pipeline:", {k: v for k, v in p.items() if k != "runs"})
