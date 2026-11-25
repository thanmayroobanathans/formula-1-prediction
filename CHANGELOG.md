# Changelog

## 2.3.0.dev16 — Phase 16: stagewise Plackett-Luce and the M4 Bayesian model
Goal: the first algorithm upgrade judged by the Phase 15 gate.
- Step 0a, embargo: the default CPCV embargo is now **10** race weekends (was 1); 0, 1 and
  3 run as sensitivity. Phase 15 showed B2's CPCV score still moving at e = 3, so 1 was
  not enough. `validation.DEFAULT_EMBARGO`, `validate --embargo`, `gate --embargo`.
- Step 0b, stagewise Plackett-Luce (`models/base.py`, `models/stagewise.py`): separate
  multipliers for places 1 | 2-3 | 4-10 | 11+, fitted out-of-fold (3 folds), Gumbel-trick
  sampler per stage, optional monotone constraint. New models "B1 stagewise",
  "M1 stagewise", "M4 stagewise". Its own family in the trial counts.
- `race_scores()` on B1, M1 and M4: scores, per-driver spread, DNF probabilities and the
  scale, so wrappers reuse a model's scores without re-deriving them.
- M4 hierarchical Bayesian car/driver model (`models/bayes.py`, PyMC, extra `bayes`):
  car random walk over 3-race blocks with winter and rule-reset jumps, driver skill
  random walk by season, learned rookie prior, Gaussian quali/practice likelihoods,
  Plackett-Luce finishing likelihood; ADVI (Adam) with a convergence check and one
  continuation, or NUTS with R-hat/divergence checks (`BayesFitError` otherwise);
  per-driver uncertainty feeds the sampler's spread.
- `pitwall-f1 strengths YEAR [--after-round N] [--plot PNG]`: car and driver strengths
  with 80% intervals; `plotting.strength_intervals`.
- Experiment log: new runs are `phase` "16"; reports read every phase >= 15.
- CI: tests install `.[dev,bayes]`; train.yml `validate-p16` matrix (3 candidates x 3
  moments, default embargo, no pools) feeding the same validation report and gate_v2.
- Docs: methods.md sections 10-11, docs/handoff/phase16.md.

## 2.2.0.dev15 — Phase 15: the validation upgrade
Goal: make it hard to fool ourselves. Every later phase's gate uses this.
- `predict/validation.py`: combinatorial purged cross-validation for races (2018-2025,
  never 2026): N = 8 time blocks, k = 2 test blocks -> 28 splits, embargo of e race
  weekends after each test block (default 1; 0, 3 and 10 also run), purge guard,
  assembly into 7 full backtest paths (López de Prado ch. 12), `check_split` raises
  `LeakageError` for any training race inside a test block or its embargo.
- `predict/significance.py`: circular block bootstrap (3 races), one-sided bootstrap
  p-value, Diebold-Mariano with the Harvey-Leybourne-Newbold correction, the deflated
  score (Deflated Sharpe Ratio on per-race improvement), PBO by combinatorially
  symmetric CV, Benjamini-Hochberg.
- `predict/experiments.py`: experiment log v2 (`experiment_id`, `family`, `params_hash`,
  `phase`, `validation`, `embargo`, `cpcv_blocks`, `cpcv_test_blocks`, `data_hash`), old
  rows stay readable, per-race losses in `reports/experiments/<id>.csv`, trial counting,
  log merging for parallel CI jobs. Content-based ids: re-running an identical
  experiment adds no row.
- `backtest.gate_v2` -> `GateDecision` (bootstrap CI on 2025 for win AND podium,
  deflated p < 0.05 once >= 10 trials, CPCV median path; DM + block bootstrap as
  context; reasons for every check). The old `gate()` is unchanged. `load_evidence`
  reads logged losses (no retraining).
- `ensemble.PooledModel` (pool as a RaceModel for CPCV, weights from inner
  out-of-fold predictions of the training races only), `pool_forward`, `pool_name`.
- `predict/validation_report.py` + `plotting/validation_plots.py`:
  `reports/validation.md` with CPCV box plots and the embargo plot, gates, trials, PBO,
  FDR, the front-of-grid diagnosis (Plackett-Luce stage temperatures).
- CLI: `pitwall-f1 validate [...] [--report-only] [--merge LOG...]`,
  `pitwall-f1 gate CANDIDATE INCUMBENT --moment M`.
- `train.yml`: `validate` matrix (one job per moment, fast models) after the dataset
  job, `validate-report` merges and commits `reports/` only. `.gitattributes`:
  union merge for `reports/experiments.csv`.
- Fix: M1/M2 strength features forecast a race dated before the end of the training
  data from a filter run only over the earlier training races (needed for CPCV;
  walk-forward unchanged). Test: `test_strength_features_forecast_past_races_from_the_past_only`.
- `CONTRIBUTING.md` (rules for phases 15-30), `docs/methods.md` (all the maths), LaTeX
  appendix paragraph. Coverage floor raised to 90%. `tests/test_phase15.py` (31 tests).
- Result (first use of the gate): see `reports/validation.md` and `lab/phase15_pools.md`.
  `reports/final_config.json` is NOT changed.

## 2.1.0.dev14 — Phase 14: quant analysis (paper trading only)
### Step 0: the after-qualifying problem (found on the real data)
The first backtest had every model losing to the plain grid after qualifying, with
B1/M1 retirement (DNF) log loss jumping from ~0.34 to ~0.77 at that moment only.
`examples/diagnose_quali.py` + `.github/workflows/diagnose.yml` compared before and
after on the real 2025 season (report: `reports/diagnose_quali.md`). Cause: extreme
inputs. At Miami 2025 the old models gave every driver a 100% retirement chance.
Fixes (each with a test in `tests/test_phase14_quali_fix.py`):
1. `quali_gap_pct` is measured inside the deepest session each driver reached (a dry
   Q1 lap is never compared with a wet Q3 pole), made monotone in qualifying order and
   capped at 5%; team-mate gap capped at ±3% (was up to 95%).
2. `practice_pace_gap_pct` capped at 5% (was up to 58%: installation laps).
3. `design_matrix` treats ±inf as missing (it used to become 1.8e308).
4. B1, M1 and M2 clip standardised inputs to ±5 sd (`base.standardise`, `Z_CLIP`).
Result on 2025: after-qualifying DNF log loss 0.767 → 0.328 (B1) and 0.767 → 0.327 (M1).
NOT fixed: for the winner, the grid baseline still beats B1/M1 after qualifying
(win LL 0.096 vs ~0.124): see the handoff notes. `reports/final_config.json` stays
frozen (it already uses the grid after qualifying).

### The market module (`pitwall/market/`)
- `odds.py`: decimal / fractional / American / probability prices, overround,
  de-vigging (multiplicative, additive, power, Shin), expected value, edge.
- `staking.py`: Kelly (single), exact multi-outcome Kelly (Smoczynski–Tomkins),
  quarter-Kelly `StakePolicy` with edge / value / odds filters and per-bet and per-race caps.
- `sources.py`: `ManualCsvSource` (prices typed in from public pages,
  `data/markets/<year>-<round>.csv`, every problem reported with its line), read-only
  `PolymarketSource` (public Gamma event + CLOB price history, no key; only for a slug you
  name), `Roster` name matching (id, code, full name, surname, accents).
- `compare.py`: market snapshot with a leakage guard (only prices seen before the
  prediction), consensus over sources, engine vs market table, log-loss comparison with a
  paired bootstrap, Benter's two-stage conditional logit (does the engine add
  information beyond the market?) with standard errors, walk-forward combination.
- `paper.py`: pretend tickets written before the race (never overwritten), settlement
  with closing line value, per-moment pretend bankrolls, ROI with bootstrap interval,
  drawdown, per-race Sharpe, risk of ruin, races needed for significance.
- `report.py`: `reports/market.md` with plots.
- CLI: `market template|check|find`, `paper trade|settle|report`.
- Workflows: paper trade after each prediction (skips without prices); settle + report
  in the Monday score job. Race PDF: an "Engine vs market" section when prices exist.
- `tests/test_phase14_market.py` (35 tests). No real money anywhere, by design.

## 2.0.0rc2 — Phase 13
- `pitwall/report/`: LaTeX race-preview PDF (Jinja2 with `\VAR{}` / `\BLOCK{}` delimiters,
  LuaLaTeX + fontspec, every data value escaped). Sections: cover, one-minute summary,
  full table vs B0, win/podium/points bars, position heatmap, team comparison, last visit's
  tyre strategy (OpenF1), safety-car rate, "why" (B1 weights and the favourite's factors),
  championship impact (expected points, title odds), track record, methodology, sources
  and licences, disclaimer. Sections degrade to "not available"; 6 pages typical, max 8.
- CLI `pitwall-f1 report YEAR EVENT --moment M [--out file.pdf] [--tex-only]`.
- `.github/workflows/race-report.yml`: Friday 06:00 / Saturday 22:00 UTC + manual; PDF to a
  GitHub Release and `docs/reports/`; tests workflow compiles the report once.
- Records now store the race context (forecast, circuit facts) the model saw.
- `tests/test_phase13.py`.

## 2.0.0rc1 — Phase 12
- `predict/live.py`: `ToppedUpHistory` (f1db + newer Jolpica races), `IdMapper`
  (Jolpica -> f1db by id, car number, team name, circuit coordinates; overrides in
  `predict/id_overrides.csv`; `IdMappingError` lists every unmatched id),
  `build_live_rows` (same dataset code for the upcoming race; entry list from qualifying
  or the last race checked against OpenF1; live Open-Meteo forecast at the moment's lead;
  `SettledWeekend` never requests a session within an hour of its end; flags + warnings
  for every missing piece; refuses finished races).
- `predict/records.py`: prediction records (never overwritten), scoring, `track_record.md`,
  the GitHub Pages site in `docs/`. `predict/service.py`: the steps behind the CLI.
- `predict/final.py`: the one final 2026 run (`final_2026.md`, `FINAL_RUN.json`, refuses an
  unfinished season, `--force` re-runs are labelled).
- `FinalPipeline.predict(..., live=True)` for unraced locked-season races; `default_config`.
- CLI: `next`, `predict`, `backtest`, `dataset build`, `score`, `final-2026`; exit codes.
- Workflows: `tests` (+ CPU torch), `predict`, `train` (10-seed matrix), `score`.
- Plots: `prediction_bars`, `position_heatmap`, `track_record_plot`. README v2.
- Fixes: weekend track temperature uses the latest practice with data; OpenF1 `drivers`
  endpoint; Open-Meteo live forecast; test fix in `test_phase9.py` (nested approx).

## 2.0.0.dev11 — Phase 11 (unreleased)
- `models/scores.py`: `ScoreSource` protocol (`scores`, `training_scores`) with adapters
  for B1, M1/M1-pure and M2 (`as_score_source`); predictions unchanged (tested).
- M3 `RaceSimulator`: structural Monte Carlo (pace from a score source with score_sd;
  grid/pace blend a_c by circuit overtaking; neutralised races with kappa_sc and two-car
  crashes; wet races from a logistic on past RaceWet; mechanical DNFs with a shared team
  shock + incidents, shrunk to field rates); parameters fitted by a mixture-of-
  Plackett-Luce likelihood on training rows; seeded per race.
- `ensemble.py`: online stacking (`stack_forward`) with equal / linear / log pools,
  softmax weights with an L2 pull to equal, `LeakageError` guard; `build_stacks`,
  `fit_weights`, `pool`, `result_from_predictions`, `choose_pipelines`, report sections.
- `calibration.py`: online matrix temperature (P^T + Sinkhorn) and Platt scaling for P_dnf
  (`calibrate_forward`, `fit_temperature`, `fit_platt`).
- `pipeline.py`: `FinalPipeline`, `PipelineConfig` / `MomentChoice`,
  `reports/final_config.json`; refuses 2026 unless `final=True`.
- Report: `extra` sections. Plots: `weights_plot`, `start_weight_plot`.
- `examples/backtest_ensemble_2025.py` (`--seeds`, `--fast`, `--reuse`; predictions cached,
  never committed); `tests/test_phase11.py`.

## 2.0.0.dev10 — Phase 10 (unreleased)
- M2 `NeuralRanker` (PyTorch CPU): linear + embeddings + MLP, guaranteed monotonicity,
  Plackett-Luce + DNF loss, time-based early stopping, seed ensemble, determinism,
  integrated gradients; `StrengthFeatures` refactor; ablation report; explanation plots;
  `nn` extra and CPU torch in CI.

## 2.0.0.dev9 — Phase 9 (unreleased)
- Dataset OBSERVATIONS; Kalman strength filter; M1 / M1-pure; `score_sd` in the Monte
  Carlo; `compare_models` and `gate`; strength plots.

## 2.0.0.dev8 — Phase 8 (unreleased)
- `RaceModel` protocol and shared probability maths; B0, B1, B2; walk-forward backtest with
  leakage guards and the 2026 lock; metrics; report; calibration plot.

## 2.0.0.dev7 — Phase 7 (unreleased)
- Free data sources (f1db, OpenF1 historical, Open-Meteo); the leakage-free dataset.

## 1.0.0 — Phase 6
- Packaging as `pitwall-f1`, CLI, CI with ruff, pytest + coverage and mypy, network smoke tests.

## 0.5.0 — Phase 5
- Telemetry from Live Timing; speed trace, delta, gear and speed maps.

## 0.4.0 — Phase 4
- Live Timing with the user's own F1 TV token; compounds, tyre life, track status.

## 0.3.0 — Phase 3
- Tyre degradation, bootstrap pace intervals, Monte Carlo title odds.

## 0.2.0 — Phase 2
- Qualifying, sprints, standings, `Season`; gallery plots; title-fight calculator.

## 0.1.0 — Phase 1
- Jolpica client with cache and rate limiting; `Session`, `Laps`; lap-time scatter.
