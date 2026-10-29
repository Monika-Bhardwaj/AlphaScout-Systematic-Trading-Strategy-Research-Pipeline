#!/usr/bin/env python
"""Run AlphaScout on real data you have downloaded. NOT exercised in the build sandbox (no network).

  NSE:     python scripts/run_real.py nse /path/to/bhavcopy_dir [--actions corporate_actions.csv]
  Binance: python scripts/run_real.py binance BTCUSDT,ETHUSDT,... 2019-01-01 2025-01-01

NSE: download the daily cash Bhavcopy files yourself (current "CM-UDiFF Common Bhavcopy Final" zips, and
legacy archives for pre-July-2024 dates) into one folder. Provide corporate actions as a CSV with columns
symbol(=ISIN if key=isin), ex_date, kind[split|bonus|dividend], price_factor, amount, announce_date.
Read docs/METHODOLOGY.md section 'Known limitations' before trusting any number."""
import sys
import warnings

warnings.filterwarnings("ignore", message="Sortedness of columns")

from alphascout.pipeline import PipelineConfig, run_pipeline

kind = sys.argv[1]
if kind == "nse":
    from alphascout.data.nse_bhavcopy import NSEBhavcopyAdapter
    actions = sys.argv[sys.argv.index("--actions") + 1] if "--actions" in sys.argv else None
    raw = NSEBhavcopyAdapter(sys.argv[2], actions_csv=actions).load()
elif kind == "binance":
    from alphascout.data.binance import BinanceAdapter
    raw = BinanceAdapter(sys.argv[2].split(","), sys.argv[3], sys.argv[4]).load()
else:
    sys.exit(__doc__)
run_pipeline(raw, PipelineConfig(), outdir=f"runs/real_{raw.name}")
