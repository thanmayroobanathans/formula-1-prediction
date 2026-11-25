# Formula 1 Prediction (pitwall-f1)

[![tests](https://github.com/thanmayroobanathans/formula-1-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/thanmayroobanathans/formula-1-prediction/actions/workflows/tests.yml)

A small, readable Formula 1 data-analysis **and race-prediction** library, inspired by [FastF1](https://github.com/theOehrly/Fast-F1) and built step by step as a learning project by a first-year Mechanical Engineering student. Every module is short, typed and documented, so you can read how each feature works.

Before every race weekend it publishes win, podium, points and DNF probabilities for every driver, made by open-source models trained only on free data, saves them in git **before** the race, and grades them afterwards: a public, timestamped track record.

> Educational project, **not betting advice**. Not affiliated with Formula 1. F1, FORMULA ONE and related marks are trademarks of Formula One Licensing B.V.

**Links:** [predictions site](https://thanmayroobanathans.github.io/formula-1-prediction/) · [track record](reports/track_record.md) · [backtest report](reports/backtest.md) · [validation report](reports/validation.md) · [methods](docs/methods.md) · [contributing](CONTRIBUTING.md) · [changelog](CHANGELOG.md)

## What it does

| Area | Highlights |
|---|---|
| **Results & laps** | Race, qualifying and sprint results; lap-by-lap times; stints; pit stops |
| **Seasons** | Standings, points per round, "who can still win the title", time-travel with `after_round` |
| **Analysis** | Tyre degradation (fuel-corrected, robust Theil–Sen fit), bootstrap pace intervals, Monte Carlo title odds |
| **Live Timing** *(needs your own F1 TV token)* | Real tyre compounds, tyre life, SC/VSC/red-flag laps |
| **Telemetry** *(needs your own F1 TV token)* | Speed traces, time delta, gear and speed track maps |
| **Plots** | 15+ matplotlib plots in a dark FastF1-style theme |
| **Race prediction** *(v2)* | Dataset at 3 moments (pre / practice / quali), 7 models, online ensemble + calibration, walk-forward backtests |
| **Live predictions** | `pitwall-f1 predict next`, records in `predictions/`, graded track record, GitHub Actions automation |
| **Race preview PDF** | A LaTeX report per race: probabilities, figures, strategy, safety car, explanations, championship impact |

## Install

Python 3.11 or newer.

```bash
pip install "pitwall-f1[ml] @ git+https://github.com/thanmayroobanathans/formula-1-prediction.git"

# the neural network (M2) needs PyTorch. On Linux install the small CPU-only wheel FIRST:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# or, for development
git clone https://github.com/thanmayroobanathans/formula-1-prediction.git
cd formula-1-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Extras: `ml` (pyarrow, scikit-learn, LightGBM, Jinja2), `nn` (PyTorch), `report` (Jinja2; the PDF also needs LuaLaTeX from a TeX distribution such as TeX Live), `dev` (everything + test tools).

The distribution is named `pitwall-f1` because `pitwall` is taken on PyPI; you still write `import pitwall`.

## Quick start

```python
import pitwall
from pitwall.plotting import driver_laptimes_scatter, use_pitwall_style

use_pitwall_style()
session = pitwall.get_session(2024, "Monza").load()        # round number or name
print(session.results[["Position", "Driver", "Team", "Points"]].head())

laps = session.laps.pick_drivers("LEC").pick_wo_box().pick_quicklaps()
driver_laptimes_scatter(session, "LEC")
```

### Seasons and the title fight
```python
season = pitwall.Season(2026).load()
print(pitwall.title_fight_summary(season))
print(pitwall.who_can_still_win(season))

from pitwall.analysis import simulate_championship
print(simulate_championship(season).head())               # chance of winning, 0–1
```

### Analysis
```python
from pitwall.analysis import degradation_table, pace_difference

print(degradation_table(session.laps.pick_drivers("LEC")))
diff = pace_difference(session.laps, "LEC", "PIA")
print(diff.difference, diff.low, diff.high, diff.is_clear)
```

### Live Timing and telemetry (your own F1 TV token)
```python
pitwall.load_live_timing(session)                           # compounds, tyre life, track status
print(session.laps.pick_green().pick_compounds("HARD").head())

quali = pitwall.get_session(2024, "Monza", "Q").load()
pitwall.load_telemetry(quali, drivers=["NOR", "PIA"])
from pitwall.plotting import speed_trace
speed_trace(quali, ["NOR", "PIA"])
```

## Race predictions (v2)

### Quick start
```bash
pitwall-f1 dataset build --from 2018          # one-time: f1db + Jolpica + OpenF1 + Open-Meteo (cached)
pitwall-f1 next                               # the next race and its latest prediction
pitwall-f1 predict 2026 Qatar --moment pre    # before the weekend
pitwall-f1 predict next --moment practice     # an hour or more after practice
pitwall-f1 predict next --moment quali --json qatar.json
pitwall-f1 report 2026 Qatar --moment quali --out qatar.pdf    # the LaTeX race preview
pitwall-f1 backtest --season 2025 --fast      # walk-forward backtest -> reports/backtest.md
pitwall-f1 score                              # grade finished predictions -> reports/track_record.md
```

```python
from pitwall.predict import FinalPipeline, load_dataset
from pitwall.predict.service import default_sources, predict_race, format_prediction

out = predict_race(2026, "Qatar", "quali", default_sources(), save=False)
print(format_prediction(out.prediction, out.teams))
```

### How a prediction is made
1. **Data** (all free): race history from f1db, topped up from Jolpica for races newer than the last f1db release; practice long-run pace, track temperature and safety-car history from OpenF1's historical API; weather forecasts from Open-Meteo.
2. **Rows at a moment**: one row per driver with only what is known at that moment (`pre`, `practice`, `quali`). Anything missing gets an availability flag of 0 and a neutral default, never a guess. The upcoming race is built by the same code as the training rows.
3. **Models**: B0 grid-slot baseline, B1 rank-ordered logit, B2 LightGBM ranker, M1 Kalman strength filter (cars and drivers), M2 monotone neural network (PyTorch, CPU), M3 race simulator. Each gives a strength score; Plackett–Luce sampling (20,000 simulated races) turns scores into position probabilities.
4. **Ensemble and calibration**: the configured models (`reports/final_config.json`, chosen by the backtest; an equal B1 + M1 blend until then) are blended with weights fitted only on earlier races, and optionally tempered so that "20%" happens about 20% of the time.
5. **Record**: `predictions/<year>-<round>-<moment>.json` with the UTC time, git hash, config hash, version, availability flags and every driver's probabilities, plus B0's at the same moment. Never overwritten.

### Results
| | Win log loss | Podium log loss |
|---|---|---|
| Target (best open-source model found) | 0.075 | 0.191 |
| pitwall backtest 2024–25, after qualifying (best: the grid baseline) | 0.111 | 0.224 |
| pitwall backtest 2024–25, before the weekend (best: B1 logistic) | 0.144 | 0.278 |
| Live 2026 forward test | see [track record](reports/track_record.md) | |

Honest reading: nothing reaches the targets yet, and after qualifying no model beats the
starting grid on its own (details and the Phase 14 fixes in `reports/diagnose_quali.md`).

The 2026 season is a **locked final test**: tuning happens on 2018–2025 only. After Abu Dhabi (6 December 2026), `pitwall-f1 final-2026 --i-understand-this-is-final` runs the frozen pipeline once and writes `reports/final_2026.md` and `reports/FINAL_RUN.json`; a second run needs `--force` and is labelled "RE-RUN — not a clean final test".

### Automation (free GitHub Actions on a public repo)
| Workflow | When (UTC) | What |
|---|---|---|
| `tests` | every push | ruff + pytest on Python 3.11–3.13 (CPU torch), LaTeX compile check, mypy |
| `predict` | Thu 12:00 (pre), Fri 22:00 (practice), Sat 22:00 (quali), manual | predicts the next race, commits the record + site |
| `race-report` | Fri 06:00 (pre), Sat 22:00 (quali), manual | builds the PDF, publishes it as a GitHub Release and on the site |
| `train` | Tue 03:17, manual | dataset, 10 M2 seeds as a job matrix, backtest of every model, validation (CPCV, one job per moment); commits `reports/` only |
| `score` | Mon 06:00, manual | grades finished races, updates the track record |

Cron jobs can start late; every job is safe to rerun. Data and models are never committed (`.gitignore`); the download cache lives in the Actions cache. To switch on: **Settings → Actions → General → Workflow permissions: Read and write**, and **Settings → Pages → Deploy from a branch: `main` / `docs`**.

## Validation: not fooling ourselves (Phase 15)

With about 48 test races, a dozen models and many settings, some "improvements" are pure
luck. Phase 15 measures that luck, and every later phase must pass its gate.

```bash
pitwall-f1 validate                          # walk-forward + CPCV (B0, B1, M1, B2), embargo 0/1/3/10,
                                             # B0+B1 / B0+M1 pools after qualifying -> reports/validation.md
pitwall-f1 validate --moments quali --sensitivity      # one moment, no extra embargoes
pitwall-f1 validate --include-m2 --m2-seeds 1          # also M2 (slow)
pitwall-f1 validate --report-only                      # rebuild the report from the log
pitwall-f1 gate "POOL-log B0+B1" "B0 grid slot" --moment quali   # from logged losses, no retraining
```

- **Walk-forward** (2024-2025, only earlier races) stays the primary test.
- **CPCV**: 2018-2025 cut into 8 time blocks, every pair of blocks as the test set (28
  splits), the races just after each test block removed from training (embargo:
  rolling-form features carry their results), stitched into 7 complete backtests. Each
  model gets a spread of 7 scores instead of one number.
- **Statistics**: paired and block bootstraps, Diebold-Mariano (HLN), the deflated score
  ("the best of 30 random tries looks good by luck"), PBO, Benjamini-Hochberg. Maths in
  [docs/methods.md](docs/methods.md).
- **gate_v2** keeps a change only if its 95% bootstrap interval on 2025 is below zero
  (win and podium), its deflated p < 0.05 once the family has >= 10 trials, and its median
  CPCV path beats the incumbent. Every experiment and its per-race losses are logged in
  `reports/experiments.csv` and `reports/experiments/`.
- Run times (2-core laptop-class CPU, f1db data): walk-forward of the four fast models
  about 5 min per moment; one CPCV run (28 splits, four models) about 4-7 min, M1 being
  most of it; the four CPCV pools after qualifying about 25 min. In GitHub Actions the
  validation runs as one job per moment.

## The engine vs the market (Phase 14, paper trading only)

**No real money, ever.** Real-money online betting is illegal in India and this is an
educational project. Phase 14 asks one honest question: *do the engine's odds contain
information that market prices do not?*

```bash
pitwall-f1 market template 2026 Abu          # a CSV to type PUBLIC prices into
pitwall-f1 market check 2026 Abu --moment quali   # margin, engine vs market, edge
pitwall-f1 paper trade 2026 Abu --moment quali    # pretend quarter-Kelly stakes
pitwall-f1 paper settle && pitwall-f1 paper report   # after the race -> reports/market.md
```

- Prices: typed in by hand from public pages (`data/markets/`), or Polymarket's public,
  key-free price history for an event you name. Only prices seen **before** a prediction
  are ever compared with it.
- Maths: the bookmaker's margin removed four ways (multiplicative, additive, power,
  Shin); Kelly staking, exact for a whole race (Smoczynski–Tomkins), used at a quarter
  with caps; Benter's two-stage model to test whether the engine adds information.
- Results: profit (pretend), ROI with a bootstrap interval, closing line value,
  drawdown, risk of ruin, and how many races until the result means anything.

## Phase 16: stagewise Plackett-Luce and a Bayesian car/driver model

- **Stagewise Plackett-Luce.** Phase 15 found that one "sharpness" setting makes the
  front of the race too flat (models gave the pole-sitter about 21% when pole won 67% of
  2025 races). Now places 1, 2-3, 4-10 and 11+ each get their own setting, learned from
  races the model did not train on.
- **M4 Bayesian model** (`pip install -e ".[ml,bayes]"`). Result = car + driver, each
  with an uncertainty: the car drifts race to race and jumps at a rule reset; a new
  driver starts from what past rookies were like, with wide uncertainty. See
  [docs/methods.md](docs/methods.md) sections 10-11.
- Both are judged by the Phase 15 gate (`reports/validation.md`); a model that does not
  pass stays out of the champion and is written up in `lab/`.

## Command line

```bash
pitwall-f1 results 2024 Monza
pitwall-f1 results 2024 Monza --kind Q
pitwall-f1 title 2026
pitwall-f1 title 2021 --after-round 21
pitwall-f1 odds 2026 --sims 50000
pitwall-f1 token          # checks your token without printing it
pitwall-f1 cache-clear
pitwall-f1 next | predict | report | backtest | dataset build | score | final-2026   # see above
pitwall-f1 validate | gate                                                           # Phase 15
pitwall-f1 strengths 2025 --after-round 10 --plot cars.png   # Phase 16: Bayesian car/driver pace
```

Exit codes: 0 done (or nothing to do with `--skip-if-*`), 1 error, 2 bad arguments, 3 data not available yet.

## Data sources and your F1 TV token

- **[Jolpica-F1](https://github.com/jolpica/jolpica-f1)** (the Ergast successor): results, laps, pit stops and standings. Free; pitwall stays under its rate limits (3 requests/s, 500/h) and caches everything.
- **[f1db](https://github.com/f1db/f1db)** (CC BY 4.0): the whole race history in one download.
- **[OpenF1](https://openf1.org)** (data CC BY-NC-SA 4.0): practice, weather, stints and race control from 2023, **historical tier only**. pitwall refuses any session that ended less than an hour ago (the live tier is paid).
- **[Open-Meteo](https://open-meteo.com)** (CC BY 4.0, free for non-commercial use): archived and current forecasts.
- **F1 Live Timing archive:** since October 2026 it requires an F1 TV Access (or higher) subscription token. pitwall reads **your own** token from the `PITWALL_F1TV_TOKEN` environment variable:

  ```bash
  export PITWALL_F1TV_TOKEN='...'     # Windows: set PITWALL_F1TV_TOKEN=...
  ```

  The token is never cached, logged or printed. On HTTP 401/403 pitwall stops and explains; it never retries around authentication or blocks, and uses no mirrors or shared tokens. Tokens expire after a few days. Keep downloaded data for personal use.

Everything except Live Timing and telemetry works without a token. **The prediction code (`pitwall.predict`) never uses the Live Timing archive or your token**, and no paid odds or data APIs are used anywhere. Because OpenF1 data is CC BY-NC-SA, the predictions are for **non-commercial** use.

## Cache

Downloads are cached in `~/.pitwall_cache` (set `PITWALL_CACHE` to move it). Past seasons are kept forever; the current season refreshes every 6 hours. Clear it with `pitwall-f1 cache-clear`.

## Project layout

```
pitwall/
├── core.py            Session and Laps (a DataFrame subclass with F1 helpers)
├── season.py          Season: schedule, results, sprints, standings
├── points.py          points systems 2010+
├── title_fight.py     "who can still win" calculator
├── timing.py          Live Timing enrichment (tyres, track status)
├── telemetry.py       car data, positions, distance, delta
├── auth.py            your F1 TV token from the environment
├── cache.py           atomic JSON disk cache
├── cli.py             the pitwall-f1 command
├── sources/           Jolpica client, Live Timing client, feed parsers
├── analysis/          degradation, bootstrap, Monte Carlo
├── predict/           dataset, models, backtest, ensemble, calibration, pipeline,
│                      live.py (top-up + live rows), records.py, service.py, final.py,
│                      validation.py (CPCV), significance.py, experiments.py (log v2)
├── report/            LaTeX race preview (template, figures, LuaLaTeX build)
└── plotting/          colours, lap/race/qualifying/season/analysis/telemetry/prediction plots
predictions/           one JSON per live prediction (committed = timestamped proof)
reports/               backtest.md, validation.md, experiments.csv (+ experiments/ per-race losses),
                       track_record.md, final_config.json
lab/                   ideas that failed the gate, with a write-up of why
docs/                  the GitHub Pages site
```

## Development

```bash
pytest                                   # offline tests (no internet, no token)
pytest --cov=pitwall                     # with coverage
pytest --run-network                     # also the real-API smoke tests
ruff check .
mypy pitwall
```

## Known limitations

- Jolpica has no tyre or track-status data; without Live Timing, stints are inferred from pit stops and SC/VSC laps are filtered by the 107% rule.
- Degradation uses an assumed fuel effect (0.05 s/lap); vary it to test how sensitive your conclusions are.
- Bootstrap intervals treat laps as independent, which makes them a little too narrow.
- The Monte Carlo model is deliberately simple (recent form, independent races, shrunk DNF rates).
- Track status is per leader lap; telemetry lap boundaries can be off by a few tenths of a second.
- No real-time streaming during sessions (that needs a SignalR Core client).
- Predictions: FIA grid penalties announced after qualifying are not parsed (the grid is the qualifying order); driver changes before qualifying are only flagged, not modelled; one race is a tiny sample, so the track record is noisy for a long time.
- Jolpica ids are mapped to f1db ids automatically (id, car number, team name, circuit coordinates); a rename that defeats this stops with a clear error and a line to add to `pitwall/predict/id_overrides.csv`.

## How to reproduce
1. `pip install -e ".[dev]"` (plus CPU torch), then `pitwall-f1 dataset build --from 2018` (about 30–60 min the first time, all cached).
2. `pitwall-f1 backtest --season 2024 --season 2025 --write-config-if-missing` (hours with M2; `--fast` for minutes).
3. Pin the f1db release for an exact rerun: `F1dbLoader(version="2026.20.0")`. Every backtest is logged in `reports/experiments.csv` with its git hash.

## Credits

- [FastF1](https://github.com/theOehrly/Fast-F1) by theOehrly and contributors: the inspiration, and the community's documentation of Live Timing feeds.
- [Jolpica-F1](https://github.com/jolpica/jolpica-f1) for keeping the Ergast-style API alive.
- [f1db](https://github.com/f1db/f1db) (CC BY 4.0), [OpenF1](https://openf1.org) (CC BY-NC-SA 4.0) and [Open-Meteo](https://open-meteo.com) (CC BY 4.0) for free data.

## Licence

MIT for the code (see `LICENSE`). Data belongs to its respective owners and keeps its own licence (above); predictions built on OpenF1 data are for non-commercial use.

**Disclaimer:** pitwall-f1 is educational. Nothing in this repository is betting or financial advice.
