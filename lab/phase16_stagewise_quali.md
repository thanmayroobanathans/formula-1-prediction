# Phase 16: stagewise Plackett-Luce after qualifying — FAILED the gate (but close)

**Question (from Phase 15).** One Plackett-Luce temperature leaves the front too flat:
pole won 67% of 2025 races, B1/M1 gave the pole-sitter about 21%. Do separate stage
multipliers (1 | 2-3 | 4-10 | 11+, fitted out-of-fold) fix it after qualifying?

**Result.** They fix most of it, but not enough to beat the grid on 2025.

| Model | WF win LL 2025 | CPCV median win LL | vs B0 on 2025 (win) |
|---|---|---|---|
| B0 grid slot (champion) | 0.0960 | 0.1190 | — |
| B1 logistic | 0.1234 | 0.1204 | worse |
| B1 stagewise | 0.1004 | 0.1018 | +0.0045 [-0.0159, +0.0258] |
| M1 stagewise | 0.0993 | 0.1017 | +0.0034 [-0.0162, +0.0235] |

Stagewise beats B0 on all 7 CPCV paths (2018-2025) by about 0.017, but on the gate
season 2025 the difference is inside the noise, and PBO for the stagewise family at
quali is 0.69 ("looks like luck"). Same pattern as the Phase 15 pools: 2025 was an
unusually pole-dominated season.

**Decision.** Champion after qualifying stays B0. Stagewise *does* replace B1 before the
weekend and after practice (passes against B1; see `reports/phase16_decision.md`).
Next idea for quali: add the grid slot itself as a feature inside the stagewise model
(the stage multipliers sharpen the front, but the base score still under-weights pole).
