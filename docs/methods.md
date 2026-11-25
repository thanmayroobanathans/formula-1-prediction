# Methods: how pitwall checks that an improvement is real (Phase 15), and the Phase 16 models

This page holds the maths behind `pitwall/predict/validation.py`,
`pitwall/predict/significance.py` and `backtest.gate_v2`. The race report's methodology
appendix links here. Plain-English summaries come first in every section.

Notation: there are $T$ races. For race $t$, $\ell^{A}_t$ is model A's loss (the mean
binary log loss over the drivers of that race, for winner or podium), and
$d_t = \ell^{A}_t - \ell^{B}_t$ the per-race difference (negative = A is better).

---

## 1. Walk-forward (the primary test)

*In words: predict every race of 2024-2025 with models trained only on earlier races.*

For each test race $t$ (in date order), models are refitted every `retrain_every` races
on the rows of the same moment with $\text{Date} < \text{Date}_t$. A training row dated on
or after the race raises `LeakageError`. The race is passed to `predict()` without its
labels. This is what really happens on a race weekend, so it stays the primary test.

## 2. Combinatorial purged cross-validation (the robustness check)

*In words: cut 2018-2025 into 8 pieces, test on every pair of pieces, and stitch the
results into 7 complete backtests. One lucky stretch of races can no longer decide.*

**Blocks and splits.** The $T$ races of 2018-2025 (never 2026) are cut, in time order,
into $N$ consecutive blocks (default $N = 8$, about 22 races each). Each choice of $k$
test blocks (default $k = 2$) is one split, so there are $\binom{N}{k} = 28$ splits.
Each split trains on the other blocks and predicts the races of its test blocks.

**Purge and embargo.** pitwall's features are rolling windows of *past* races: average
finish over 5 races, DNF rates over 10, championship points so far. A training race
that comes just *after* a test block therefore carries the test races' results inside
its features, and a model that learns from it learns a little about the test labels.
The **embargo** removes the $e$ race weekends after every test block from training
(default $e = 10$ since Phase 16; the sensitivity check also runs $e = 0, 1, 3$). Why 10:
the longest look-back of any form feature is 10 races (DNF rates), so a race more than 10
races after a test race no longer has that test race's result inside any of its
features. Phase 15 measured the cost of a too-short gap: B2 (LightGBM) scored better
with $e = 0$ than with $e = 10$ in 7 of 8 model-moments, a small but consistent sign
that it was using the test results hidden in later races' features. The **purge** removes training
races whose date falls inside a test block's date span; with one-race labels it is empty,
but it is kept as a guard. `check_split` raises `LeakageError` if a training race is a
test race, lies within the embargo after one, or is dated inside a test block.

Formally, with races indexed $0..T-1$ in time order and $S$ the test set of a split,
$$\text{train} = \{\, j \notin S : \nexists\, i \in S \text{ with } 0 < j - i \le e \,\}.$$

**Paths.** Each block is tested in $\binom{N-1}{k-1}$ splits. Ordering those splits for
every block, path $p$ takes block $b$'s prediction from the $p$-th split that tests $b$.
There are
$$\varphi = \frac{k}{N}\binom{N}{k} = \binom{N-1}{k-1} = 7$$
paths (López de Prado, *Advances in Financial Machine Learning*, ch. 12), and each path
predicts every race exactly once with a model that never trained on that race's block.
A model's CPCV result is the distribution of its 7 path log losses (mean, 5th, 50th and
95th percentile, interpolated).

**Models that track state.** CPCV trains on races *after* the test race. A Kalman
filter (M1, and M2's strength inputs) is a running summary of the past, so its forecast
for a test race is taken from a filter run only over the training races *before* that
race (Phase 15 fix in `strength_features.py`). The other models are plain regressions
and need no change.

**Pools in CPCV.** A pool's weights need out-of-sample predictions. In walk-forward they
come from earlier races (`stack_forward`). In CPCV there are no "earlier races", so
`PooledModel` cuts the split's *training* races into 3 consecutive folds, fits the
members on the other folds (with a 1-race embargo after the held-out fold), predicts the
held-out fold, and fits the weights on those inner out-of-fold predictions.

## 3. Bootstraps

*In words: replay history many times by drawing races at random, and see how much the
average difference wobbles.*

**Paired bootstrap** (`metrics.paired_bootstrap`, Phase 8). Draw $T$ races with
replacement, keep each race's pair $(\ell^A_t, \ell^B_t)$ together, compute $\bar d^{*}$.
Repeat $B = 10\,000$ times; the 2.5th and 97.5th percentiles of $\bar d^{*}$ are the 95% CI.

**Block bootstrap** (`significance.block_bootstrap`). Neighbouring races can be alike
(the same upgrade package, the same part of the season). The circular moving-block
bootstrap draws $\lceil T/L \rceil$ random starting races and takes $L$ consecutive races
from each (wrapping round), with $L = 3$ by default. Dependence inside a block is kept,
so the interval is wider when neighbouring races are correlated.

**One-sided bootstrap p-value** (used for the false-discovery rate):
$p = \big(1 + \#\{\bar d^{*} \ge 0\}\big)/(B + 1)$. A 95% CI entirely below 0 corresponds
to $p < 0.025$.

## 4. Diebold-Mariano test with the Harvey-Leybourne-Newbold correction

*In words: a t-test on the per-race differences, a second opinion to the bootstrap.*

$$\mathrm{DM} = \frac{\bar d}{\sqrt{\hat V/T}},\qquad
\hat V = \hat\gamma_0 + 2\sum_{j=1}^{h-1}\hat\gamma_j,\qquad
\hat\gamma_j = \frac1T\sum_{t=j+1}^{T}(d_t-\bar d)(d_{t-j}-\bar d).$$
Each race is a one-step forecast, so $h = 1$ and $\hat V = \hat\gamma_0$. With few races
Harvey, Leybourne and Newbold (1997) use
$$\mathrm{DM}^{*} = \mathrm{DM}\,\sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}
\;\sim\; t_{T-1}.$$
pitwall reports the one-sided p-value $P(t_{T-1} \le \mathrm{DM}^*)$ (A better) and the
two-sided one.

## 5. The deflated score

*In words: "the best of 30 random tries looks good by luck". If we try many settings and
keep the best, the best one will look better than it really is. The deflated score asks
whether the winner beats what the luckiest of N useless settings would have shown.*

Adapted from the Deflated Sharpe Ratio (Bailey & López de Prado 2014). Let
$x_t = \ell^{\text{old}}_t - \ell^{\text{new}}_t$ (positive = better) and
$$\widehat{SR} = \frac{\bar x}{s_x}\quad\text{(a per-race "Sharpe ratio")}.$$

**Expected maximum of N useless tries.** If $N$ configurations all have true score 0 and
their estimated scores have standard deviation $\sigma_{SR}$, the expected best is
$$SR_0 = \sigma_{SR}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac1N\right)
+ \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right],\qquad \gamma \approx 0.5772.$$
For $N = 1$, $SR_0 = 0$. By default $\sigma_{SR} = \sqrt{1/(T-1)}$, the spread of a
useless configuration's estimated score. Trials of one family are usually correlated, so
this is conservative (it overstates the luck).

**Deflated probability** (probabilistic Sharpe ratio against $SR_0$, with skewness
$\hat\gamma_3$ and ordinary kurtosis $\hat\gamma_4$ of $x$):
$$\mathrm{DSR} = \Phi\!\left(\frac{(\widehat{SR} - SR_0)\sqrt{T-1}}
{\sqrt{1 - \hat\gamma_3\widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}}\right),
\qquad p_{\text{deflated}} = 1 - \mathrm{DSR}.$$

**Number of trials.** $N$ = number of distinct configurations (`params_hash`, fitted
values such as Kalman variances excluded) of the candidate's family at that moment in
`reports/experiments.csv`, whatever validation or dataset they ran on. The gate requires
$p_{\text{deflated}} < 0.05$ once $N \ge 10$.

## 6. Probability of Backtest Overfitting (PBO)

*In words: split the races in two halves in every possible way. Pick the configuration
that looks best on one half. How often is it below the median on the other half? 50% =
the "best" is just luck; near 0% = a real winner.*

Combinatorially symmetric cross-validation (Bailey, Borwein, López de Prado & Zhu 2017).
Take the $T \times M$ matrix of per-race losses of $M$ configurations (common races, time
order) and cut its rows into $S$ consecutive blocks ($S = 8$ by default, because pitwall
has few races). For every choice $c$ of $S/2$ blocks as in-sample:

1. $n^{*} = \arg\min_m$ in-sample mean loss;
2. $\omega_c = \operatorname{rank}(n^{*}) / (M + 1)$ out-of-sample, rank 1 = worst;
3. $\lambda_c = \log\big(\omega_c / (1-\omega_c)\big)$.

$$\mathrm{PBO} = \frac{1}{\binom{S}{S/2}}\sum_c \mathbf 1[\lambda_c < 0]$$
(a tie at exactly the median counts one half).

## 7. Benjamini-Hochberg false discovery rate

*In words: if you run 13 gates, one may pass by luck. BH keeps the expected share of
lucky passes among all passes at most $q$.*

Sort the $m$ p-values $p_{(1)} \le \dots \le p_{(m)}$; reject $H_{(1)},\dots,H_{(k)}$ with
$k = \max\{i : p_{(i)} \le q\,i/m\}$ ($q = 0.05$). Adjusted p-values:
$\tilde p_{(i)} = \min_{j \ge i}\min(1, p_{(j)}\,m/j)$.

## 8. The gate (`backtest.gate_v2`)

| Check | Required? |
|---|---|
| Paired-bootstrap 95% CI of (candidate − incumbent) on 2025 entirely below 0, win AND podium log loss | yes |
| Deflated p < 0.05 (win log loss) | yes, once the family has ≥ 10 trials |
| Median CPCV path win log loss below the incumbent's (same races, same blocks) | yes |
| Diebold-Mariano (HLN), block bootstrap (3 races) | context |

The gate reads only the logged per-race losses (`reports/experiments/<id>.csv`), so any
later phase can re-run it without retraining: `pitwall-f1 gate CANDIDATE INCUMBENT
--moment M`.

## 9. Plackett-Luce stage temperatures (the after-qualifying diagnosis)

*In words: Plackett-Luce picks the winner, then 2nd from the rest, and so on, all with
one "sharpness" setting. If the setting that fits the winner best is much sharper than
the one that fits the midfield, a single setting makes the front too flat: it spreads
win probability away from the favourites.*

For scores $s_i$ and a temperature $\beta$, place $k$ in a race is chosen among the
drivers still left $R_k$ with probability
$\exp(\beta s_{(k)}) / \sum_{j \in R_k} \exp(\beta s_j)$. For a group of places $G$
(1; 2-3; 4-10; 11+) pitwall finds
$$\hat\beta_G = \arg\max_\beta \sum_{\text{races}}\sum_{k \in G}
\Big[\beta s_{(k)} - \log\!\sum_{j \in R_k} e^{\beta s_j}\Big]$$
on out-of-sample scores (models fitted on 2018-2023, scored on 2024-2025). $\hat\beta = 1$
means the model's own scale is right for those places.

## 10. Stagewise Plackett-Luce (Phase 16, step 0)

*In words: section 9 showed one sharpness setting cannot fit both the fight for the win
and the midfield. Stagewise Plackett-Luce gives each group of places its own setting,
learned from races the model did not train on.*

Places are split into stages $G_1 = \{1\}$, $G_2 = \{2, 3\}$, $G_3 = \{4..10\}$,
$G_4 = \{11+\}$. With multipliers $m_1..m_4 > 0$, place $k \in G_g$ is filled with
probability $\exp(m_g s_{(k)}) / \sum_{j \in R_k} \exp(m_g s_j)$. With all $m_g = 1$ this
is ordinary Plackett-Luce. The multipliers are fitted by maximum likelihood (L-BFGS-B on
$\log m_g$, bounds 0.05-20) on **out-of-fold** scores: the base model (B1, M1 or M4) is
refitted on 3 folds of the training races (every 3rd race in date order held out), so
every training race is scored by a model that never saw it. Without that the in-sample
scores are too confident and the fitted $m_1$ is biased down. Optional monotone
constraint ($m_1 \ge m_2 \ge \dots$) via softplus increments.

Sampling uses the Gumbel trick per stage: for the places of stage $g$, add Gumbel noise to
$m_g s_i$ over the drivers still unplaced and take the top $|G_g|$. Retirements are drawn
first and always fill the last places, as in the plain sampler.

Stagewise PL is **its own family** in the trial counts (deflated score, PBO, BH), so its
trials do not dilute or borrow from B1 or M1.

## 11. M4: hierarchical Bayesian car and driver model (Phase 16)

*In words: each race result is car + driver. The car's speed drifts a little from race to
race, can jump over the winter and jumps a lot when the rules reset (2014, 2017, 2022,
2026). A driver's skill drifts slowly from season to season. A driver nobody has seen gets
the typical rookie's skill with wide uncertainty. Every number comes with an interval.*

Pace (percent gap to the field, lower = faster) of driver $d$ in team $t$ at race $r$:
$$\text{pace}_{d,r} = c_{t,b(r)} + a_{d,y(r)}$$

- Car, random walk over blocks $b$ of 3 races (blocks never cross seasons):
  $c_{t,b} = c_{t,b-1} + \varepsilon$, $\varepsilon \sim N(0, \sigma_c^2 + \mathbb 1_{\text{new
  season}}\sigma_w^2 + \mathbb 1_{\text{rule reset}}\sigma_e^2)$, with
  $\sigma_c \sim HN(0.2)$, $\sigma_w \sim HN(0.5)$, $\sigma_e \sim HN(1)$.
- Driver skill, random walk by season: $a_{d,y} = a_{d,y-1} + N(0, \sigma_a^2)$,
  $\sigma_a \sim HN(0.2)$. First season: veterans (already in the data before the window)
  $N(0, \sigma_v^2)$; rookies $N(\mu_r, \sigma_r^2)$, $\mu_r \sim N(0.2, 0.3)$,
  $\sigma_r \sim HN(0.5)$: the rookie prior is *learned* from past rookies.
- Likelihoods: qualifying gap $\sim N(\text{pace}, \sigma_q^2)$, practice gap
  $\sim N(\text{pace}, \sigma_p^2)$, finishing order $\sim$ Plackett-Luce with scores
  $-\beta \cdot \text{pace}$, $\beta \sim HN(3)$ (finishers only; DNFs are a separate
  logistic model on the usual moment features, as in B1).
- Inference: ADVI (mean-field, Adam, 15,000 iterations; one continuation if the loss is
  still falling; `BayesFitError` on NaN or no convergence). NUTS is available
  (`method="nuts"`, R-hat <= 1.05 and zero divergences or it refuses).

Forecasting race $r^*$: posterior draws are carried forward — car steps for the races
since the last block ($\approx$ days / 42 steps), plus winter (and era) variance in a new
season; an unknown team gets $N(0, \text{NEW\_CAR\_VAR})$, an unknown driver the rookie
prior. At the practice and quali moments the race's own centred gaps are then used as
one Gaussian observation each (conjugate update of mean and variance) — the same
information B1 gets as features, so the comparison is fair. The prediction is
Plackett-Luce with mean score $-\beta\,\bar{\text{pace}}$ and per-driver spread
$\beta\,\text{sd}(\text{pace})$ (the `score_sd` path of the sampler): uncertain drivers
(rookies, a new era) get wider finishing distributions, which a single score cannot do.

`pitwall-f1 strengths 2025 --after-round 10 --plot cars.png` prints the car and driver
tables with 80% intervals.

## References

- M. López de Prado (2018), *Advances in Financial Machine Learning*, Wiley, ch. 7 and 12.
- D. Bailey, M. López de Prado (2014), "The Deflated Sharpe Ratio", *Journal of Portfolio
  Management* 40(5).
- D. Bailey, J. Borwein, M. López de Prado, Q. J. Zhu (2017), "The Probability of Backtest
  Overfitting", *Journal of Computational Finance* 20(4).
- F. Diebold, R. Mariano (1995), "Comparing Predictive Accuracy", *JBES* 13(3).
- D. Harvey, S. Leybourne, P. Newbold (1997), "Testing the equality of prediction mean
  squared errors", *International Journal of Forecasting* 13(2).
- R. Plackett (1975), "The analysis of permutations", *Applied Statistics* 24(2);
  R. D. Luce (1959), *Individual Choice Behavior*.
- A. Kucukelbir et al. (2017), "Automatic Differentiation Variational Inference", *JMLR* 18.
- A. Gelman et al. (2013), *Bayesian Data Analysis*, 3rd ed., ch. 5 (hierarchical models).
- H. Künsch (1989), "The jackknife and the bootstrap for general stationary
  observations", *Annals of Statistics* 17(3) (block bootstrap).
- Y. Benjamini, Y. Hochberg (1995), "Controlling the False Discovery Rate", *JRSS B* 57(1).
