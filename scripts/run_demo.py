#!/usr/bin/env python
"""Run the full pipeline on a synthetic world with known ground truth.

  python scripts/run_demo.py --world planted --out runs/planted    # real edge exists: expect survivors
  python scripts/run_demo.py --world noise   --out runs/noise      # no edge: expect NO survivors
"""
import argparse
import warnings

warnings.filterwarnings("ignore", message="Sortedness of columns")

from alphascout.data.synthetic import generate_synthetic, noise_world
from alphascout.pipeline import PipelineConfig, run_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--world", choices=["planted", "noise"], default="planted")
ap.add_argument("--assets", type=int, default=100)
ap.add_argument("--days", type=int, default=2500)
ap.add_argument("--candidates", type=int, default=None, help="default: the 104-candidate grid")
ap.add_argument("--momentum", type=float, default=2.0, help="planted momentum strength (planted world)")
ap.add_argument("--pairs", type=int, default=6, help="planted cointegrated pairs (planted world)")
ap.add_argument("--seed", type=int, default=21)
ap.add_argument("--out", default="runs/demo")
a = ap.parse_args()

raw = (noise_world(a.assets, a.days, a.seed) if a.world == "noise" else
       generate_synthetic(a.assets, a.days, a.seed, planted_pairs=a.pairs, planted_momentum=a.momentum))
res = run_pipeline(raw, PipelineConfig(n_candidates=a.candidates), outdir=a.out)
print("\nTiming:", {k: round(v, 3) for k, v in res["timing"].items()})
print(f"Wrote {a.out}/summary.json, candidates.csv, rejection_log.csv, tearsheet_*.html")
