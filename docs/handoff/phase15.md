# Phase 15 handoff notes

Saved from the Phase 15 delivery (DELIVERY.md) so the repo holds everything the next phase needs.

## Result in one paragraph
No candidate passes `gate_v2`. After qualifying, the four B0 + B1 / B0 + M1 pools beat the grid on
all 7 CPCV paths over 2018-2025 (median win LL about 0.114 vs 0.1185), but on 2025 they are clearly
worse (win LL +0.009 to +0.010, 95% CI above 0), and with 12 pool configurations tried the deflated
p is 0.999. Keep B0 after qualifying; `reports/final_config.json` is unchanged. Why B1/M1 lose on the
winner: Plackett-Luce's single temperature is too flat at the front (best out-of-sample temperature
about 2.0 for 1st place vs about 1.1 for 4th-10th and 0.2 for 11th+); in 2025 pole won 67% of races
while B1/M1 gave the pole-sitter about 21%. Data caveat: the real-data run was done on an f1db-only
dataset (OpenF1 and Open-Meteo were unreachable from the build sandbox). B0 after qualifying matches
Phase 14 exactly (0.1111), but B1/M1/B2 lack practice and weather features. The weekly `train`
workflow regenerates everything on the full dataset.
## Fixes to existing code, each with the test that covers it
| Fix | File | Test |
|---|---|---|
| A Kalman strength forecast for a race dated before the end of the training data now comes from a filter run only over the earlier training races (before, CPCV would have used the *latest* car strengths for a 2019 race). Walk-forward is unchanged. | `pitwall/predict/strength_features.py` | `test_phase15.py::test_strength_features_forecast_past_races_from_the_past_only` (+ all Phase 9-11 M1/M2 tests unchanged) |
| `log_experiments` writes the v2 columns and per-race loss files; an old-format log is upgraded in place with old rows kept readable; an identical re-run adds no row | `pitwall/predict/backtest.py` | `test_phase15.py::test_log_round_trip_and_old_rows_stay_readable`, `test_phase8.py::test_run_backtest_logs_compares_and_reports` |
| `BacktestResult` gains an optional `data_hashes` field (positional construction still works) | `pitwall/predict/backtest.py` | `test_phase9.py::test_gate_keeps_only_clear_improvements`, `test_phase15.py::test_old_gate_still_works` |
| Coverage floor raised from 75% to 90% | `pyproject.toml` | the whole suite: 287 passed, 93.9% |
| `train.yml`: new `validate` matrix and `validate-report` jobs | `.github/workflows/train.yml` | `test_phase12.py::test_workflows_parse_and_stay_free[train.yml]` |
| LaTeX methodology appendix: one paragraph on the new gate, link to `docs/methods.md` | `race_report.tex` | `test_phase13.py` render tests (PDF compile skipped here: no LuaLaTeX fonts in the sandbox; the CI `latex` job checks it is <= 8 pages) |

## Handoff notes
**Done:** CPCV with purge + embargo and 7 paths; block bootstrap, DM-HLN, deflated score, PBO, BH;
experiment log v2 with per-race losses and trial counting; `gate_v2` (old `gate()` kept); CLI
`validate` / `gate`; CI matrix; `reports/validation.md` on real data; the pool experiment and the
front-of-grid diagnosis; `lab/` write-up; 31 new tests (ruff clean, 287 passed, coverage 93.9%).

**Not done / known limits:**
- The real-data report uses an f1db-only dataset (no OpenF1 practice or safety-car data, no weather), so
  `practice` equals `pre` for B1/M1/B2. Rerun `pitwall-f1 validate` on the full dataset (the Tuesday
  workflow does this) before reading the B1/M1 numbers as final.
- Practice was run at embargo 1 only (to save time); `pre` and `quali` have embargo 0/1/3/10. Run times
  in the report cover the quali job only.
- Embargo: linear and Kalman models move by at most 0.0005 win LL across embargo 0-10, B2 by 0.0044;
  the score is lower with embargo 0 than 10 for 7 of 8 model-moments (a small but consistent leakage
  signature). The report therefore recommends a default embargo of 10; the code default stays 1 in this
  phase as planned.
- Deflated score uses the conservative null spread sqrt(1/(T-1)); correlated trials overstate the luck.
- PBO uses 8 blocks (few races); with 43 common races it is coarse.
- M2 was not run in CPCV (`--include-m2` exists, untested on real data for time).
- Per-race loss files add about 3.6 MB to `reports/experiments/` (content-based ids, so weekly reruns
  on unchanged data add nothing).
- mypy: the new modules are clean; 6 older errors remain (informational job).

**Phase 16 starts with:** set the CPCV default embargo to 10 (or justify 1), rerun `validate` on the full
dataset, then attack the front-of-grid problem as one family: a place-dependent (stagewise)
Plackett-Luce temperature, gated with `gate_v2` against B0 after qualifying.
