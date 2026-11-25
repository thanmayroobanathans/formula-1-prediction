# Phase 16 decision: candidates gated against the real champion

`reports/validation.md` gates every model against B0 (the grid). But the frozen
`reports/final_config.json` uses **B1 logistic** before the weekend and after practice,
and **B0** only after qualifying. So the decision gates are:

| Moment | Champion | M1 stagewise | B1 stagewise | M4 Bayesian |
|---|---|---|---|---|
| pre | B1 logistic | **PASS** (win -0.0143 [-0.0213, -0.0073]; podium -0.0266; deflated p 0.003; CPCV 7/7) | fail (win CI touches 0) | fail (worse on every check) |
| practice | B1 logistic | **PASS** (win -0.0146 [-0.0214, -0.0078]; podium -0.0267; deflated p 0.001; CPCV 7/7) | fail (win CI touches 0) | fail |
| quali | B0 grid slot | fail (win +0.0034 [-0.0162, +0.0235] on 2025; CPCV 7/7 better) | fail (same pattern) | fail |

Run: train #2 on branch `phase-16` (Actions run 45701720452, 2026-11-25), default embargo 10.

## Recommendation for `reports/final_config.json` (frozen: not changed here)
- **pre** and **practice**: replace `B1 logistic` with `M1 stagewise`. It is the first
  candidate in this project to pass gate_v2 against the champion, with every context
  check passing too (Diebold-Mariano p < 0.001, block bootstrap, deflated p < 0.01).
- **quali**: keep `B0 grid slot`. Stagewise PL closes most of the Phase 15 gap
  (2025 win LL 0.0993 vs B1's 0.1234; B0 0.0960) and beats B0 on all 7 CPCV paths, but
  on 2025 alone the difference is noise, and PBO for stagewise at quali is 0.69.

## Caveat
The pre and practice numbers are almost identical for every model: the dataset used
by CI has little practice data, so "practice" is close to "pre" here. The practice
result says "no worse than pre", not "practice data helps".

## Full gate output
```
B1 stagewise does NOT pass the gate against B1 logistic at 'pre' (failed: win LL: 95% CI on 2025 entirely below 0). Keep the incumbent; move the candidate to lab/.

[FAIL, required] win LL: 95% CI on 2025 entirely below 0: -0.0085 [-0.0179, +0.0017] over 24 races
[PASS, required] podium LL: 95% CI on 2025 entirely below 0: -0.0198 [-0.0317, -0.0058] over 24 races
[FAIL, context] win LL: deflated p < 0.05: per-race score +0.337 over 24 races; the best of 3 useless tries would reach about +0.178 by luck; deflated p = 0.248 (required once >= 10 trials; now 3)
[PASS, required] win LL: CPCV median path better: median 0.1166 vs 0.1292; better on 7 of 7 paths
[FAIL, context] win LL: Diebold-Mariano (HLN): statistic -1.65, one-sided p = 0.056 (two-sided 0.112), 24 races
[FAIL, context] win LL: block bootstrap (3 races): -0.0085 [-0.0175, +0.0006]

Trials in family 'stagewise PL' at 'pre': 3.
M1 stagewise PASSES the gate against B1 logistic at 'pre': every required check passed.

[PASS, required] win LL: 95% CI on 2025 entirely below 0: -0.0143 [-0.0213, -0.0073] over 24 races
[PASS, required] podium LL: 95% CI on 2025 entirely below 0: -0.0266 [-0.0345, -0.0185] over 24 races
[PASS, context] win LL: deflated p < 0.05: per-race score +0.792 over 24 races; the best of 3 useless tries would reach about +0.178 by luck; deflated p = 0.003 (required once >= 10 trials; now 3)
[PASS, required] win LL: CPCV median path better: median 0.1157 vs 0.1292; better on 7 of 7 paths
[PASS, context] win LL: Diebold-Mariano (HLN): statistic -3.88, one-sided p = 0.000 (two-sided 0.001), 24 races
[PASS, context] win LL: block bootstrap (3 races): -0.0143 [-0.0204, -0.0085]

Trials in family 'stagewise PL' at 'pre': 3.
M4 Bayesian does NOT pass the gate against B1 logistic at 'pre' (failed: win LL: 95% CI on 2025 entirely below 0; podium LL: 95% CI on 2025 entirely below 0; win LL: CPCV median path better). Keep the incumbent; move the candidate to lab/.

[FAIL, required] win LL: 95% CI on 2025 entirely below 0: +0.0074 [-0.0003, +0.0157] over 24 races
[FAIL, required] podium LL: 95% CI on 2025 entirely below 0: +0.0230 [+0.0048, +0.0411] over 24 races
[FAIL, context] win LL: deflated p < 0.05: per-race score -0.359 over 24 races; the best of 2 useless tries would reach about +0.108 by luck; deflated p = 0.991 (required once >= 10 trials; now 2)
[FAIL, required] win LL: CPCV median path better: median 0.1687 vs 0.1292; better on 0 of 7 paths
[FAIL, context] win LL: Diebold-Mariano (HLN): statistic +1.76, one-sided p = 0.954 (two-sided 0.092), 24 races
[FAIL, context] win LL: block bootstrap (3 races): +0.0074 [-0.0024, +0.0186]

Trials in family 'M4' at 'pre': 2.
B1 stagewise does NOT pass the gate against B1 logistic at 'practice' (failed: win LL: 95% CI on 2025 entirely below 0). Keep the incumbent; move the candidate to lab/.

[FAIL, required] win LL: 95% CI on 2025 entirely below 0: -0.0086 [-0.0180, +0.0015] over 24 races
[PASS, required] podium LL: 95% CI on 2025 entirely below 0: -0.0198 [-0.0320, -0.0057] over 24 races
[FAIL, context] win LL: deflated p < 0.05: per-race score +0.343 over 24 races; the best of 3 useless tries would reach about +0.178 by luck; deflated p = 0.242 (required once >= 10 trials; now 3)
[PASS, required] win LL: CPCV median path better: median 0.1165 vs 0.1292; better on 7 of 7 paths
[FAIL, context] win LL: Diebold-Mariano (HLN): statistic -1.68, one-sided p = 0.053 (two-sided 0.106), 24 races
[FAIL, context] win LL: block bootstrap (3 races): -0.0086 [-0.0176, +0.0004]

Trials in family 'stagewise PL' at 'practice': 3.
M1 stagewise PASSES the gate against B1 logistic at 'practice': every required check passed.

[PASS, required] win LL: 95% CI on 2025 entirely below 0: -0.0146 [-0.0214, -0.0078] over 24 races
[PASS, required] podium LL: 95% CI on 2025 entirely below 0: -0.0267 [-0.0348, -0.0185] over 24 races
[PASS, context] win LL: deflated p < 0.05: per-race score +0.833 over 24 races; the best of 3 useless tries would reach about +0.178 by luck; deflated p = 0.001 (required once >= 10 trials; now 3)
[PASS, required] win LL: CPCV median path better: median 0.1155 vs 0.1292; better on 7 of 7 paths
[PASS, context] win LL: Diebold-Mariano (HLN): statistic -4.08, one-sided p = 0.000 (two-sided 0.000), 24 races
[PASS, context] win LL: block bootstrap (3 races): -0.0146 [-0.0207, -0.0088]

Trials in family 'stagewise PL' at 'practice': 3.
M4 Bayesian does NOT pass the gate against B1 logistic at 'practice' (failed: win LL: 95% CI on 2025 entirely below 0; podium LL: 95% CI on 2025 entirely below 0; win LL: CPCV median path better). Keep the incumbent; move the candidate to lab/.

[FAIL, required] win LL: 95% CI on 2025 entirely below 0: +0.0078 [-0.0000, +0.0162] over 24 races
[FAIL, required] podium LL: 95% CI on 2025 entirely below 0: +0.0232 [+0.0047, +0.0413] over 24 races
[FAIL, context] win LL: deflated p < 0.05: per-race score -0.373 over 24 races; the best of 2 useless tries would reach about +0.108 by luck; deflated p = 0.992 (required once >= 10 trials; now 2)
[FAIL, required] win LL: CPCV median path better: median 0.1687 vs 0.1292; better on 0 of 7 paths
[FAIL, context] win LL: Diebold-Mariano (HLN): statistic +1.83, one-sided p = 0.960 (two-sided 0.080), 24 races
[FAIL, context] win LL: block bootstrap (3 races): +0.0078 [-0.0021, +0.0192]

Trials in family 'M4' at 'practice': 2.
B1 stagewise does NOT pass the gate against B0 grid slot at 'quali' (failed: win LL: 95% CI on 2025 entirely below 0; podium LL: 95% CI on 2025 entirely below 0). Keep the incumbent; move the candidate to lab/.

[FAIL, required] win LL: 95% CI on 2025 entirely below 0: +0.0045 [-0.0159, +0.0258] over 24 races
[FAIL, required] podium LL: 95% CI on 2025 entirely below 0: -0.0018 [-0.0277, +0.0268] over 24 races
[FAIL, context] win LL: deflated p < 0.05: per-race score -0.083 over 24 races; the best of 3 useless tries would reach about +0.178 by luck; deflated p = 0.895 (required once >= 10 trials; now 3)
[PASS, required] win LL: CPCV median path better: median 0.1018 vs 0.1190; better on 7 of 7 paths
[FAIL, context] win LL: Diebold-Mariano (HLN): statistic +0.41, one-sided p = 0.655 (two-sided 0.689), 24 races
[FAIL, context] win LL: block bootstrap (3 races): +0.0045 [-0.0195, +0.0267]

Trials in family 'stagewise PL' at 'quali': 3.
M1 stagewise does NOT pass the gate against B0 grid slot at 'quali' (failed: win LL: 95% CI on 2025 entirely below 0; podium LL: 95% CI on 2025 entirely below 0). Keep the incumbent; move the candidate to lab/.

[FAIL, required] win LL: 95% CI on 2025 entirely below 0: +0.0034 [-0.0162, +0.0235] over 24 races
[FAIL, required] podium LL: 95% CI on 2025 entirely below 0: -0.0030 [-0.0269, +0.0230] over 24 races
[FAIL, context] win LL: deflated p < 0.05: per-race score -0.065 over 24 races; the best of 3 useless tries would reach about +0.178 by luck; deflated p = 0.876 (required once >= 10 trials; now 3)
[PASS, required] win LL: CPCV median path better: median 0.1017 vs 0.1190; better on 7 of 7 paths
[FAIL, context] win LL: Diebold-Mariano (HLN): statistic +0.32, one-sided p = 0.623 (two-sided 0.754), 24 races
[FAIL, context] win LL: block bootstrap (3 races): +0.0034 [-0.0199, +0.0244]

Trials in family 'stagewise PL' at 'quali': 3.
```
