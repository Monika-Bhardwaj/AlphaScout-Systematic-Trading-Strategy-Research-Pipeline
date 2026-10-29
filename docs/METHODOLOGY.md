# AlphaScout: methodology and overfitting-control procedure

All numbers below come from **synthetic markets with known ground truth** produced by this repo
(`data/synthetic.py`). No real-market result exists yet. Real-data caveats are in "Known limitations".

## The claim, and the evidence behind each part

| Claim | Mechanism | Evidence |
|---|---|---|
| Features and strategies never see the future | PIT store with `as_of`; returns from raw prices + ex-date actions; one engine-level execution lag; truncation audit that **stops the pipeline** | `test_leakage_detector...`, `test_engine_enforces_one_bar_execution_lag`, `test_pipeline_refuses_to_run_with_a_leaky_strategy` |
| Survivorship bias is removed | universe = whatever had a bar that date, delisted names kept, terminal delisting return applied | `test_universe_is_survivorship_free_and_matches_snapshot`, `test_survivor_only_universe_is_upward_biased` |
| Bad data cannot masquerade as signal | validation + rejection log; spikes, non-positive/inconsistent bars, unadjusted splits | all injected ticks caught; unrecorded splits quarantined (`test_data.py`, `test_bad_tick_burst_is_caught_by_validation`) |
| The winner is judged against luck | every candidate counted as a trial; PBO, Deflated Sharpe, Romano-Wolf over all N | noise experiments below |
| The framework is itself tested | pure-noise markets must yield no survivors; a planted edge must be found | noise experiments below |
| Survivors are stress-tested | 2x/3x costs, latency, regimes, risk controls under shocks, untouched hold-out | planted-world replay table below |

## 1. Look-ahead control

* **PIT store.** Each price row has `as_of` (bar date + publication time); each corporate action has
  `announce_date`. `build_panel(as_of=t)` sees only rows knowable at t and re-runs validation on that slice, so
  the last visible bar is never "confirmed" with its own future (the spike rule uses the next bar).
* **Returns from raw prices.** `ret_t = (close_t / f_t + div_t) / close_prev - 1`, with split factor `f` and
  dividend applied on the ex-date. No back-adjusted level series exists, so a future split cannot leak into a
  past price level. Verified against true returns: every mismatch sits next to a logged rejection.
* **Execution lag in one place.** Strategies emit weights decided at the close of row t; `engine.py` alone makes
  row t+1 earn them. Test: a book peeking at t+1 earns Sharpe > 20, a same-row book earns about 0.
* **Truncation audit** (`features/leakage.py`). For any `f(panel)`, `f(full)[:n]` must equal `f(first n rows)`.
  Anything touching the future (shift(-1), global normalisation, centred windows, global ranks) diverges and the
  report names the first bad row. Run on every feature and one strategy per family before screening.
  *Does not catch:* leakage through the data itself (e.g. a vendor back-filling), or look-ahead in how a human
  chooses which strategy to try after seeing results (see "untracked trials").

## 2. Survivorship and data quality

* `verify_universe_snapshot` compares the store's universe on a date with a known snapshot (ground truth in the
  synthetic world). Dropping delisted names flatters the equal-weight return in the test.
* Rules: duplicates; non-positive/null prices; negative volume; OHLC inconsistency; spike-and-revert (>30% one-bar
  move that returns within 10%); unadjusted split/consolidation (price ratio near 1/2, 1/3, 1/4, 1/5, 1/10, 2, 3,
  5, 10 with no recorded action within 3 days: the *return* is quarantined, not fed to strategies).
* Bug found by the ground-truth test: an unadjusted split appears at `canonical_ratio * (1 + that day's true
  return)`, so a 3% tolerance missed a 5:1 split that fell on a -6% day. Tolerance is now 12%. Trade-off: a
  genuine ~-50% move with no action would be quarantined.

## 3. Overfitting control (the part that matters)

**Trials.** The generator fixes N before any result is seen (104 by default). Candidates that never trade still
count as trials (`n_trials`) even though they have no Sharpe.

**PBO via CSCV** (Bailey, Borwein, Lopez de Prado, Zhu). The T x N net-return matrix is cut into S=16 slices; for
each of the 12,870 ways to pick 8 slices as in-sample, the in-sample Sharpe winner's *relative OOS rank* is
recorded; PBO is the fraction of combinations where it falls in the bottom half. It describes the **selection
procedure over the whole set**, not one strategy. For skill-less strategies it should be about 0.5.

**Deflated Sharpe.** SR0 is the expected maximum Sharpe among N skill-less trials; DSR is the probability a
strategy's true Sharpe exceeds SR0 given sample length, skew and kurtosis. SR0 scales with the null standard
deviation of a Sharpe estimate, and that choice matters:

| Null variance | Planted-edge world (N=104, best raw Sharpe 2.40) |
|---|---|
| cross-section of the N observed Sharpes (the paper's practical estimator) | SR0 = **2.98**, DSR passes **0** |
| theoretical sampling variance 1/(T-1) | SR0 = **0.90**, DSR passes 18 |

The cross-section estimator measures real skill dispersion when the candidate set is heterogeneous (momentum about
+1.7, costly mean-reversion about -1.1 after costs), which inflates SR0 and removes power. The pipeline gate uses
the sampling null, which assumes roughly uncorrelated returns, **and** the Reality Check below, which does not.

**Romano-Wolf stepdown Reality Check.** Each strategy's returns are centred (null: zero mean), time-rows are
resampled in stationary blocks (mean length 20, preserving serial and cross-strategy dependence), and a strategy is
significant only if its Sharpe beats the bootstrap distribution of the maximum over the strategies still in
play. Controls family-wise error across correlated, heterogeneous candidates. Measured FWER on noise: 5% at
nominal 5%.

**Gate.** A strategy survives iff DSR(sampling) >= 0.95 **and** Romano-Wolf p <= 0.05 **and** set-level
PBO <= 0.40. The thresholds are choices, not theorems. BH-FDR q-values are reported as a cross-check.

**Selection-procedure evidence.** Walk-forward (train 504, gap 5, test 63) and purged/embargoed 5-fold CV
(label overlap purged, embargo 5 rows) both answer "if I had to pick the best candidate from allowed training rows
only, how does the pick do on rows it never saw?". They do not replace the correction.

**Hold-out.** The last 20% of dates is touched once, after the gate, for finalists only.

## 4. Does the framework catch its own weaknesses? (`scripts/run_noise_experiments.py`)

**iid noise**, 200 strategies, T=1500, 60 repeats per row:

| True Sharpe of one planted strategy | mean PBO | naive: cherry-picked best looks significant | corrected gate passes anything |
|---|---|---|---|
| 0 (pure noise) | 0.503 | 100% | **0 / 60** |
| 1.0 | 0.426 | 100% | 1.7% |
| 2.0 | 0.142 | 100% | 61.7% |
| 3.0 | 0.007 | 100% | 100% |

Student-t(4) noise gives the same picture (0 / 60 false positives; detection 5%, 65%, 97% at Sharpe 1, 2, 3).

**Full pipeline on markets with no tradable structure** (zero market drift, no pre-delisting drift, no planted
edge; real generators, costs, leakage audit, all gates), 20 worlds: at least one candidate looks significant
*uncorrected* in **40%** of worlds; the final gate passes **0 / 20**; mean PBO 0.45. Sample run
(`runs/noise`): best raw Sharpe 0.57 vs SR0 0.90; nothing significant; the diagnostic top candidate's hold-out
Sharpe was **-0.98**.

**Power is limited, by design.** With ~200 tries over about 6 years, a true Sharpe of 1.0 is statistically
indistinguishable from luck. A bar that high is the price of not fooling yourself; more data or fewer trials
lowers it.

**Planted-edge world** (`runs/planted`: 100 assets, 2500 days, planted momentum + 6 cointegrated pairs): 18 of 104
survive, **all momentum**; mean-reversion and pairs correctly pass nothing. PBO 0.000.
Hold-out Sharpe of survivors 1.5 to 2.5 for the strongest (research 1.9 to 2.4): the edge persists but shrinks
(winner's curse; a unit test asserts the shrinkage).

| Finalist | research Sharpe | hold-out (vectorised) | replay, 1x costs | 3x costs | latency +2 bars |
|---|---|---|---|---|---|
| mom_L250_s5_h5_q0.2 | 2.40 | 2.38 | 2.05 | 0.68 | 2.16 |
| mom_L120_s5_h21_q0.2 | 2.28 | 2.52 | 2.26 | 1.10 | 2.07 |
| mom_L60_s5_h5_q0.2 | 1.89 | 1.86 | 1.37 | -1.00 | 1.74 |

Cost stress separates the candidates: the 5-day-rebalance variants collapse at 3x costs; 21-day variants hold up.
Latency barely matters for slow signals on daily bars (expected; it would not for intraday).

## 5. Execution realism and speed

* Screen: linear costs per unit turnover (default 15 bps one-way, plus a borrow/financing proxy on shorts).
* Event simulator (`backtest/event_sim.py`): delayed orders, fill at next-bar VWAP, participation cap (10% of lagged
  ADV, partial fills carry over), spread + fees + square-root impact, borrow, per-name/gross caps, name freeze on
  untrusted prints, daily-loss halt, drawdown halt. Positions are tracked as notional grown by total return, so
  splits and dividends never touch fills.
* **Timing** (100 assets x 2500 days, 104 candidates, single core): leakage audit 10.2 s; screen 13.7 s, of which
  weight generation 13.3 s and vectorised PnL accounting 0.34 s. The vectorised accounting is **about 32x faster**
  than the same accounting written as a per-day/per-asset loop (identical to 1e-17). One event-simulator run costs
  about 17x a vectorised accounting pass. **Weight generation (Python loops) dominates the screen, so the 32x
  applies to accounting only**; moving the strategy loops to numba is the real next speed-up.

## 6. Adversarial robustness (`robustness/shocks.py`, `tests/test_shocks_and_event_sim.py`)

| Shock | Observed |
|---|---|
| Adversarial -20% gap on the book's longs | daily-loss halt fires |
| 5-day bear raid (-15%/day on longs) | equity 0.85 with controls vs 0.69 without |
| Flash crash, -50% on 10% of names | name freeze fires for exactly the hit names |
| Factor reversal (winners crash, losers rally, common selloff) | drawdown halt fires; 0.92 with controls vs 0.68 without |
| Generic correlated regime break (common factor + 2.5-3x idiosyncratic vol) | **no control fired**: a dollar-neutral book is insulated from common shocks |
| Bad-tick burst, 40 corrupted bars | at least 95% caught (all in the test run) |
| 2% of bars deleted | panel, weights and PnL stay finite |

Controls limit damage **after** a gap; they cannot undo the gap itself.

## Known limitations

1. **No real-data results.** NSE Bhavcopy and Binance adapters were tested only on fixtures I wrote from the
   documented layouts; no real file was parsed. NSE files contain no corporate actions or delisting returns:
   supply an actions CSV; delisting returns default to an assumption.
2. **Synthetic worlds encode my choices.** Planted momentum gives a research Sharpe near 2.4, far above what real
   markets offer; it proves the framework can find an edge, not that one exists. Planted pairs were **not**
   found: 252-day windows give weak power against a scan of thousands of pairs.
3. **Costs are assumptions.** Taxes, spreads, impact coefficient and borrow are placeholders. As far as I know,
   shorting cash-equity delivery is restricted in India, so long/short results should be read as
   futures-implemented.
4. **Untracked trials.** The choice of three strategy families and their grids is a researcher degree of freedom
   not counted in N. Editing the generator after seeing results adds trials the deflation cannot see.
5. **Daily bars only.** VWAP fills and impact are modelled, not observed; no intraday path.
6. **Gate assumptions.** The sampling-null DSR assumes roughly uncorrelated returns; effective N (correlated
   candidates) is not estimated, which is conservative. alpha = 5% and PBO <= 0.40 are conventions.
7. **One hold-out.** It is clean only if touched once. Re-using it contaminates it.
8. **Bad-tick rule** uses the next bar (fine offline); a live system must quarantine a print for one bar.
9. **Evidence strength.** Pipeline-level false-positive evidence is 0/20 worlds (95% upper bound about 14%); the
   stronger evidence is the 120 iid runs (0 false positives).
