# AlphaScout

Strategy research where the two failure modes that invalidate most backtests, **look-ahead bias** and
**data-snooping / backtest overfitting**, are enforced by code and tested against ground truth, not
promised in a write-up.

> The question this repo is built to answer: *"how do you know this isn't just noise?"*
> Short answer: every candidate counts as a trial, the winner is judged against what luck alone would
> produce from that many trials, and the whole procedure is itself tested on markets with no edge.

## Status (what is real, what is not)

| Area | State |
|---|---|
| Point-in-time store, survivorship-free universe, corporate actions from raw prices, bad-tick/unadjusted-split rejection log | built, tested vs synthetic ground truth |
| Features: returns/vol/momentum, volume imbalance, VWAP deviation, rolling Hurst, vectorised Engle-Granger | built, tested (EG matches statsmodels to 1e-14) |
| Automated leakage detector (truncation audit); pipeline **refuses to run** if any feature/strategy leaks | built, tested (catches `shift(-1)`, full-sample z-score, demeaning) |
| 104-candidate generator: cross-sectional momentum, per-asset mean reversion (optional Hurst gate), cointegration pairs | built |
| Vectorised screen + event-driven simulator (fills, partial fills, latency, impact, risk controls) + paper-trading replay on an untouched hold-out | built, tested |
| Walk-forward, purged/embargoed K-fold, **PBO (CSCV)**, **Deflated Sharpe**, **Romano-Wolf Reality Check**, BH-FDR | built, tested |
| Noise experiments (iid noise, fat tails, full pipeline on no-edge markets) with a planted-edge positive control | built; numbers in `docs/METHODOLOGY.md` |
| Shock tests: flash-crash gaps, adversarial bear raid, factor-reversal regime break, bad-tick burst, missing bars | built, tested |
| Tearsheets (HTML), cost stress 2x/3x, regime breakdown, vol-targeted risk | built |
| Pluggable data adapters: synthetic, NSE Bhavcopy, Binance klines | synthetic: tested. **NSE and Binance: parsers tested only on fixtures I wrote from the documented layouts; never run against real downloads (no network in the build sandbox)** |
| **Real-market results** | **none yet.** Everything reported so far is on synthetic markets with known truth. |

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                                              # 40+ tests, ~1.5 min
python scripts/run_demo.py --world planted --out runs/planted   # real edge exists  -> survivors
python scripts/run_demo.py --world noise   --out runs/noise     # no edge           -> zero survivors
python scripts/run_noise_experiments.py 6              # calibration of the whole gate -> docs/noise_results.json
```

Real data (on your machine; see the docstring in `scripts/run_real.py` and the adapters):

```bash
python scripts/run_real.py nse /path/to/bhavcopy_dir --actions corporate_actions.csv
python scripts/run_real.py binance BTCUSDT,ETHUSDT,SOLUSDT 2019-01-01 2025-01-01
```

## How the pieces fit

```
adapter -> RawMarketData -> PITStore (as_of timestamps, validation, returns from RAW prices + actions)
        -> Panel (T x A arrays) -> leakage audit (truncation test; hard stop)
        -> N candidates x vectorised screen  -> net-of-cost returns matrix (T x N), research window only
        -> gate: PBO (CSCV) + Deflated Sharpe + Romano-Wolf FWER   (all N, not just the winner)
        -> walk-forward + purged-CV evidence about the selection procedure
        -> finalists: cost stress 1/2/3x, regimes, vol-targeted risk, event-driven replay on the
           never-touched hold-out (costs 1/2/3x, latency +1/+2 bars), tearsheet
```

Design rules that matter:

* **One place applies the execution lag.** Strategies emit target weights decided at the close of row t;
  the engine alone makes row t+1 earn them. A strategy cannot get timing wrong on its own, and a test
  proves the engine pays a foresighted book enormously (Sharpe > 20) and a same-row book nothing.
* **Returns come from raw prices plus corporate actions on their ex-dates**, never from back-adjusted
  price levels, so a future split cannot leak into a past price level.
* **The number of trials is fixed before looking at results** (`strategies/generator.py`) and carried
  into the deflation.
* **The hold-out (last 20% by default) is touched once**, after the gate, for finalists only.

## Layout

```
src/alphascout/
  data/        raw.py panel.py pit_store.py validation.py synthetic.py nse_bhavcopy.py binance.py
  features/    core.py hurst.py cointegration.py leakage.py cache.py
  strategies/  momentum.py mean_reversion.py pairs.py generator.py base.py
  backtest/    engine.py (vectorised + naive reference) event_sim.py costs.py
  validation/  pbo.py dsr.py reality_check.py walkforward.py (incl. PurgedKFold) stats.py
  risk/        metrics.py sizing.py regimes.py
  robustness/  noise_test.py shocks.py
  report/      tearsheet.py
  pipeline.py
tests/  scripts/  docs/METHODOLOGY.md  docs/noise_results.json  notebooks/  runs/ (sample outputs)
```

Read `docs/METHODOLOGY.md` before trusting any number, in particular "Known limitations".
