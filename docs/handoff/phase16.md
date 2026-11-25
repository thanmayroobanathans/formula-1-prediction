# Phase 16 handoff

## What was built
- Embargo default 10 (sensitivity 0, 1, 3). Reason: Phase 15's B2 CPCV score kept moving
  between e = 1 and e = 3.
- Stagewise Plackett-Luce (family "stagewise PL"): `StagewiseModel(base_factory, name)`
  wraps any model with `race_scores()`; out-of-fold multipliers for 1 | 2-3 | 4-10 | 11+.
- M4 `BayesianModel` (PyMC, ADVI by default). `pitwall-f1 strengths` prints its tables.
- Experiment log phase "16"; the report reads phases >= 15.

## How it is validated
`train.yml` -> `validate-p16`: one job per (candidate, moment), walk-forward 2024-2025
and CPCV at e = 10, logs merged by `validate-report` into `reports/validation.md`, which
runs gate_v2 for every candidate against the champion (`reports/final_config.json`,
frozen: changes are recommended, never written).

## Decision rule
Keep a candidate only if gate_v2 passes (win AND podium CI on 2025 below 0, deflated
p < 0.05 once >= 10 trials, CPCV median path better). Otherwise write it up in `lab/`
with the numbers and the reason, and leave the champion alone.

## Known costs and limits
- M4 fit: ~1-3 min per retrain on the full 2018-2025 data (ADVI 15,000 iterations).
  "M4 stagewise" refits M4 four times per retrain, so it is not in the CI matrix; run it
  only if M4 itself passes.
- ADVI is mean-field: intervals are somewhat too narrow. NUTS (`method="nuts"`) is the
  check, at ~10x the time.
- Synthetic smoke test: M4 was between B0 and B1/M1 on win log-loss; stagewise B1 was
  slightly worse than B1 (the synthetic data has no front-of-grid effect). Real data
  decides.

## For Phase 17 (HMM)
`race_scores()` is the interface: an HMM regime model only has to return `RaceScores`
to be wrapped by `StagewiseModel` and gated the same way.
