# pitwall backtest

Seasons tested: **2024, 2025**. Walk-forward: every race is predicted by models trained only on earlier races, retrained every 4 races. Generated 2026-11-25 (git f000b05).

- LL = binary log loss averaged over driver rows (comparable to the 0.075 / 0.191 targets). Winner LL = mean of -log P_win of the actual winner. Lower is better for LL, Brier, RPS and ECE; higher for Spearman, Winner hit and Podium overlap.
- Baseline: **B0 grid slot**. At the 'quali' moment it is the grid baseline.

## Moment: pre

| Model | Win LL | Podium LL | Points LL | DNF LL | Winner LL | Win Brier | RPS | Spearman | Winner hit | Podium overlap | Win ECE | Races |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 grid slot (baseline) | 0.1473 | 0.2783 | 0.5458 | 0.3293 | 2.0062 | 0.0433 | 0.1337 | 0.5658 | 0.2708 | 0.5417 | 0.0163 | 48 |
| B1 logistic | 0.1439 | 0.2778 | 0.5132 | 0.3297 | 1.9767 | 0.0407 | 0.1280 | 0.6073 | 0.3125 | 0.6181 | 0.0167 | 48 |
| M1 strength | 0.1442 | 0.2790 | 0.5017 | 0.3294 | 1.9826 | 0.0406 | 0.1258 | 0.6194 | 0.3333 | 0.5903 | 0.0196 | 48 |
| Target: best open model | 0.0750 | 0.1910 | — | — | — | — | — | — | — | — | — | — |

Difference vs baseline (model minus baseline; negative = better), mean per race with a 95% paired-bootstrap interval:

| Model | Win LL | Podium LL |
|---|---|---|
| B1 logistic | -0.0034 [-0.0137, +0.0066] (no clear difference) | -0.0004 [-0.0108, +0.0095] (no clear difference) |
| M1 strength | -0.0031 [-0.0147, +0.0081] (no clear difference) | +0.0007 [-0.0115, +0.0124] (no clear difference) |

- B0 grid slot: win LL 0.1473 does NOT beat the 0.075 target; podium LL 0.2783 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1439 does NOT beat the 0.075 target; podium LL 0.2778 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1442 does NOT beat the 0.075 target; podium LL 0.2790 does NOT beat the 0.191 target.

## Moment: practice

| Model | Win LL | Podium LL | Points LL | DNF LL | Winner LL | Win Brier | RPS | Spearman | Winner hit | Podium overlap | Win ECE | Races |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 grid slot (baseline) | 0.2044 | 0.4301 | 0.6937 | 0.3261 | 3.0877 | 0.0491 | 0.1746 | 0.2188 | 0.1250 | 0.2500 | 0.0220 | 48 |
| B1 logistic | 0.1439 | 0.2783 | 0.5155 | 0.3295 | 1.9766 | 0.0407 | 0.1284 | 0.5959 | 0.3125 | 0.5903 | 0.0157 | 48 |
| M1 strength | 0.1442 | 0.2791 | 0.5043 | 0.3296 | 1.9829 | 0.0407 | 0.1262 | 0.6167 | 0.3542 | 0.5694 | 0.0203 | 48 |
| Target: best open model | 0.0750 | 0.1910 | — | — | — | — | — | — | — | — | — | — |

Difference vs baseline (model minus baseline; negative = better), mean per race with a 95% paired-bootstrap interval:

| Model | Win LL | Podium LL |
|---|---|---|
| B1 logistic | -0.0605 [-0.0795, -0.0404] (better) | -0.1517 [-0.1824, -0.1205] (better) |
| M1 strength | -0.0602 [-0.0788, -0.0407] (better) | -0.1510 [-0.1788, -0.1222] (better) |

- B0 grid slot: win LL 0.2044 does NOT beat the 0.075 target; podium LL 0.4301 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1439 does NOT beat the 0.075 target; podium LL 0.2783 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1442 does NOT beat the 0.075 target; podium LL 0.2791 does NOT beat the 0.191 target.

## Moment: quali

| Model | Win LL | Podium LL | Points LL | DNF LL | Winner LL | Win Brier | RPS | Spearman | Winner hit | Podium overlap | Win ECE | Races |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 grid slot (baseline) | 0.1111 | 0.2238 | 0.4809 | 0.3228 | 1.4423 | 0.0317 | 0.1149 | 0.6837 | 0.5625 | 0.7083 | 0.0136 | 48 |
| B1 logistic | 0.1298 | 0.2451 | 0.4498 | 0.3233 | 1.7307 | 0.0378 | 0.1127 | 0.7054 | 0.4167 | 0.6944 | 0.0309 | 48 |
| M1 strength | 0.1301 | 0.2456 | 0.4477 | 0.3230 | 1.7355 | 0.0378 | 0.1122 | 0.7091 | 0.4375 | 0.6875 | 0.0302 | 48 |
| Target: best open model | 0.0750 | 0.1910 | — | — | — | — | — | — | — | — | — | — |

Difference vs baseline (model minus baseline; negative = better), mean per race with a 95% paired-bootstrap interval:

| Model | Win LL | Podium LL |
|---|---|---|
| B1 logistic | +0.0187 [+0.0043, +0.0324] (worse) | +0.0213 [+0.0058, +0.0370] (worse) |
| M1 strength | +0.0190 [+0.0044, +0.0327] (worse) | +0.0218 [+0.0063, +0.0372] (worse) |

- B0 grid slot: win LL 0.1111 does NOT beat the 0.075 target; podium LL 0.2238 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1298 does NOT beat the 0.075 target; podium LL 0.2451 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1301 does NOT beat the 0.075 target; podium LL 0.2456 does NOT beat the 0.191 target.

## Gate: B1 logistic

Candidate minus the best previous model (chosen per moment by win log loss), 95% paired-bootstrap interval:

| Moment | Best previous | Win LL | Podium LL | Both clearly better? |
|---|---|---|---|---|
| pre | B0 grid slot | -0.0034 [-0.0137, +0.0066] (no clear difference) | -0.0004 [-0.0108, +0.0095] (no clear difference) | no |
| practice | B0 grid slot | -0.0605 [-0.0795, -0.0404] (better) | -0.1517 [-0.1824, -0.1205] (better) | yes |
| quali | B0 grid slot | +0.0187 [+0.0043, +0.0324] (worse) | +0.0213 [+0.0058, +0.0370] (worse) | no |

**Decision:** B1 logistic is KEPT: clearly better than the best previous model on both win and podium log loss at: practice.

## Gate: M1 strength

Candidate minus the best previous model (chosen per moment by win log loss), 95% paired-bootstrap interval:

| Moment | Best previous | Win LL | Podium LL | Both clearly better? |
|---|---|---|---|---|
| pre | B0 grid slot | -0.0031 [-0.0147, +0.0081] (no clear difference) | +0.0007 [-0.0115, +0.0124] (no clear difference) | no |
| practice | B0 grid slot | -0.0602 [-0.0788, -0.0407] (better) | -0.1510 [-0.1788, -0.1222] (better) | yes |
| quali | B0 grid slot | +0.0190 [+0.0044, +0.0327] (worse) | +0.0218 [+0.0063, +0.0372] (worse) | no |

**Decision:** M1 strength is KEPT: clearly better than the best previous model on both win and podium log loss at: practice.

## Verdicts

- B0 grid slot: win LL 0.1473 does NOT beat the 0.075 target; podium LL 0.2783 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1439 does NOT beat the 0.075 target; podium LL 0.2778 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1442 does NOT beat the 0.075 target; podium LL 0.2790 does NOT beat the 0.191 target.
- B0 grid slot: win LL 0.2044 does NOT beat the 0.075 target; podium LL 0.4301 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1439 does NOT beat the 0.075 target; podium LL 0.2783 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1442 does NOT beat the 0.075 target; podium LL 0.2791 does NOT beat the 0.191 target.
- B0 grid slot: win LL 0.1111 does NOT beat the 0.075 target; podium LL 0.2238 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1298 does NOT beat the 0.075 target; podium LL 0.2451 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1301 does NOT beat the 0.075 target; podium LL 0.2456 does NOT beat the 0.191 target.

