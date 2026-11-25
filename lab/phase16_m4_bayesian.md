# Phase 16: M4 hierarchical Bayesian car/driver model — FAILED the gate

**Question.** Does a Bayesian car + driver model (car random walk with rule-reset jumps,
driver skill random walk, learned rookie prior, Plackett-Luce finishing likelihood, ADVI)
beat the champion?

**Result (real data, run 45701720452, embargo 10).** No, clearly not, at every moment.

| Moment | M4 WF win LL 2025 | Champion WF win LL 2025 | M4 CPCV median | Champion CPCV median | Paths better |
|---|---|---|---|---|---|
| pre | 0.1489 | 0.1415 (B1) | 0.1687 | 0.1292 (B1) | 0 / 7 |
| practice | 0.1491 | 0.1413 (B1) | 0.1687 | 0.1292 (B1) | 0 / 7 |
| quali | 0.1468 | 0.0960 (B0) | 0.1627 | 0.1190 (B0) | 0 / 7 |

Against B0 after qualifying: +0.0508 win LL [+0.0300, +0.0702], Diebold-Mariano +4.79.

**Reading.** M4's losses sit between a uniform guess (win LL about 0.20 with 20 cars,
podium about 0.42) and the other models (about 0.12 / 0.26): its predictions are too
**flat**. It worked on the synthetic data, so the problem is how it meets real data.
Likely causes, in the order to check:
1. *Double-counted uncertainty.* The sampler adds the posterior spread of pace on top of
   Plackett-Luce's own Gumbel noise. If the posterior spread is wide (mean-field ADVI,
   a new-season jump at every winter), every driver's distribution widens and the
   favourites lose probability. Test: predict with `spread=None`.
2. *beta too small.* The finishing likelihood is a `Potential` on top of Gaussian gap
   likelihoods with thousands of observations; beta (the sharpness) may be pulled low.
   Test: log `hyper_["beta"]` per fit; refit beta out-of-fold (as stagewise does).
3. *Unit mismatch.* Pace is in percent of lap time; a 1% gap is huge in F1. Check that
   the quali/practice gap observations and the car steps use the same scale.
4. *ADVI quality.* Run one split with NUTS and compare beta and the spreads.

**Decision.** Champion unchanged. M4 stays in the code (it is useful for
`pitwall-f1 strengths` and as an uncertainty source) but is not used for predictions.
A follow-up (Phase 16b, or Lab L-track) should run the four checks above on one CPCV
split before any re-gate; each re-run counts as a new M4 trial.
