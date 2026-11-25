"""Phase 15: run the validation (walk-forward + CPCV + embargo sensitivity + the
B0 + B1 / B0 + M1 pools after qualifying) and write reports/validation.md.

run_validation() trains and logs; write_validation_report() reads ONLY the experiment
log and the per-race loss files, so the report (and every gate in it) can be rebuilt
without retraining: `pitwall-f1 validate --report-only`.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import logsumexp

from ..errors import DataNotAvailableError
from .backtest import (
    EXPERIMENTS_PATH, LOG_COLUMNS, REPORTS_DIR, BacktestResult, Evidence, GateDecision,
    experiment_meta, gate_v2, git_hash, load_evidence, log_experiments, moment_data_hash,
    run_backtest, strip_outcomes, summarise_by_model,
)
from .ensemble import L2_TO_EQUAL, PooledModel, pool_forward, pool_name, pool_params
from .experiments import (
    append_rows, count_trials, is_validated_phase, losses_dir_for, model_family, read_log,
    read_race_losses,
    trials_table, write_race_losses,
)
from .metrics import METRIC_NAMES, weighted_loss
from .models.base import finish_tensor
from .report import markdown_table
from .significance import benjamini_hochberg, probability_of_backtest_overfitting
from .validation import (
    CPCV_FIRST_YEAR, CPCV_LAST_YEAR, DEFAULT_BLOCKS, DEFAULT_EMBARGO, DEFAULT_TEST_BLOCKS,
    SENSITIVITY_EMBARGOES, CPCVResult, distribution, path_losses, plan_cpcv, run_cpcv,
)

VALIDATION_REPORT = REPORTS_DIR / "validation.md"
TIMES_PATH = REPORTS_DIR / "validation_times.csv"
FAST_VALIDATE_MODELS = ["B0 grid slot", "B1 logistic", "M1 strength", "B2 LightGBM"]
INCUMBENT = "B0 grid slot"
POOL_MOMENT = "quali"
POOL_PARTNERS = ["B1 logistic", "M1 strength"]
POOL_METHODS = ("linear", "log")
POOL_L2_GRID = (0.01, L2_TO_EQUAL, 0.2)
WF_SEASONS = (2024, 2025)
EMBARGO_TOLERANCE = 0.002   # win LL: "moves a lot" (about 1/6 of the B1-vs-grid gap)
STAGE_GROUPS = ((1, 1), (2, 3), (4, 10), (11, 30))
Progress = Callable[[str], None]


@dataclass
class ValidateOptions:
    """What `pitwall-f1 validate` runs."""

    models: list[str] = field(default_factory=lambda: list(FAST_VALIDATE_MODELS))
    moments: list[str] = field(default_factory=lambda: ["pre", "practice", "quali"])
    blocks: int = DEFAULT_BLOCKS
    test_blocks: int = DEFAULT_TEST_BLOCKS
    embargo: int = DEFAULT_EMBARGO
    sensitivity: tuple[int, ...] = SENSITIVITY_EMBARGOES
    include_m2: bool = False
    m2_seeds: int = 1
    pools: bool = True
    walk_forward: bool = True
    retrain_every: int = 2
    log_path: Path = EXPERIMENTS_PATH
    report_path: Path | None = VALIDATION_REPORT
    times_path: Path = TIMES_PATH


@dataclass
class LoggedConfig:
    """A name + configuration, for logging predictions that no single model made."""

    name: str
    config: dict[str, Any]

    def params(self) -> dict[str, Any]:
        """The configuration (as RaceModel.params)."""
        return dict(self.config)


# ---------- models ----------
def model_factories_for(names: Sequence[str], include_m2: bool = False,
                        m2_seeds: int = 1) -> dict[str, Callable[[], Any]]:
    """Fresh-model factories for the validation (M2 only when asked, 1-2 seeds)."""
    from .models import NeuralRanker
    from .pipeline import model_factories
    known = model_factories(n_seeds=m2_seeds)
    chosen = [n for n in names if n != "M2-full"]
    unknown = [n for n in chosen if n not in known]
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Known: {', '.join(known)}.")
    out = {n: known[n] for n in chosen}
    if include_m2:
        out["M2-full"] = lambda: NeuralRanker(n_seeds=m2_seeds, max_epochs=100,
                                              name="M2-full")
    return out


# ---------- running ----------
def run_validation(data: pd.DataFrame, options: ValidateOptions,
                   progress: Progress = print) -> list[str]:
    """Walk-forward 2024-2025, CPCV at the default embargo and the sensitivity embargoes,
    the pools after qualifying, then the report. Returns log lines (what ran, how long)."""
    factories = model_factories_for(options.models, options.include_m2, options.m2_seeds)
    lines: list[str] = []
    times: list[dict[str, Any]] = []
    wf_preds = None
    if options.walk_forward:
        progress("Walk-forward 2024-2025 ...")
        started = dt.datetime.now()
        result = run_backtest([f() for f in factories.values()], data, seasons=WF_SEASONS,
                              moments=options.moments, retrain_every=options.retrain_every,
                              log_path=options.log_path)
        wf_preds = result.predictions
        seconds = (dt.datetime.now() - started).total_seconds()
        times.append({"moment": " ".join(options.moments), "model": "all",
                      "validation": "walk_forward", "embargo": "", "seconds": seconds})
        lines.append(f"walk-forward: {seconds:.0f} s")
        if options.pools and POOL_MOMENT in options.moments:
            lines += log_walk_forward_pools(wf_preds, data, options.log_path,
                                            options.retrain_every)
    embargoes = [options.embargo] + [e for e in options.sensitivity if e != options.embargo]
    for moment in options.moments:
        for embargo in embargoes:
            progress(f"CPCV {moment}, embargo {embargo} ...")
            plan = plan_cpcv(data, options.blocks, options.test_blocks, embargo)
            res = run_cpcv([f() for f in factories.values()], data, moment, plan, progress)
            log_cpcv(options.log_path, res, [f() for f in factories.values()], data)
            for name, sec in res.seconds.items():
                times.append({"moment": moment, "model": name, "validation": "cpcv",
                              "embargo": embargo, "seconds": sec})
            lines.append(f"CPCV {moment} embargo {embargo}: "
                         f"{sum(res.seconds.values()):.0f} s")
    if options.pools and POOL_MOMENT in options.moments:
        progress("CPCV pools after qualifying ...")
        pools = default_pools()
        plan = plan_cpcv(data, options.blocks, options.test_blocks, options.embargo)
        res = run_cpcv(pools, data, POOL_MOMENT, plan, progress)
        log_cpcv(options.log_path, res, pools, data)
        for name, sec in res.seconds.items():
            times.append({"moment": POOL_MOMENT, "model": name, "validation": "cpcv",
                          "embargo": options.embargo, "seconds": sec})
        lines.append(f"CPCV pools: {sum(res.seconds.values()):.0f} s")
    write_times(options.times_path, times)
    if wf_preds is not None and POOL_MOMENT in options.moments:
        save_front_of_grid(wf_preds, data, Path(options.log_path).parent)
    if options.report_path is not None:
        path = write_validation_report(options.log_path, options.report_path,
                                       options.embargo, times_path=options.times_path)
        lines.append(f"Wrote {path}")
    return lines


def log_walk_forward_pools(preds: pd.DataFrame, data: pd.DataFrame, log_path: Path,
                           retrain_every: int) -> list[str]:
    """B0 + partner pools (linear / log, three l2 values) at the quali moment, weights
    fitted on earlier races only (warm-up 2024, gate season 2025), logged as trials."""
    needed = {INCUMBENT, *POOL_PARTNERS}
    have = set(preds.loc[preds["Moment"] == POOL_MOMENT, "Model"])
    if not needed <= have:
        return [f"pools skipped: walk-forward lacks {sorted(needed - have)}"]
    tables, configs = [], []
    for partner in POOL_PARTNERS:
        members = [INCUMBENT, partner]
        for method in POOL_METHODS:
            for l2 in POOL_L2_GRID:
                table, _ = pool_forward(preds, POOL_MOMENT, members, method,
                                        eval_seasons=[WF_SEASONS[-1]],
                                        warmup_seasons=list(WF_SEASONS[:-1]), l2=l2)
                if table.empty:
                    continue
                tables.append(table.drop(columns="Eval"))
                configs.append(LoggedConfig(pool_name(members, method, l2),
                                            pool_params(members, method, l2)))
    if not tables:
        return ["pools skipped: too few races"]
    pooled = pd.concat(tables, ignore_index=True)
    result = BacktestResult(pooled, summarise_by_model(pooled), pd.DataFrame(), WF_SEASONS,
                            retrain_every, INCUMBENT, git_hash(),
                            {POOL_MOMENT: moment_data_hash(data, POOL_MOMENT)})
    log_experiments(Path(log_path), result, configs)
    return [f"walk-forward pools: {len(configs)} configurations logged"]


def default_pools() -> list[PooledModel]:
    """The four default-l2 pools for CPCV (linear / log x B0+B1 / B0+M1). Their weights
    are fitted on inner out-of-fold predictions of each split's training races; members
    are fitted once per set of rows and shared between the pools."""
    from .pipeline import model_factories
    known = model_factories(n_seeds=1)
    cache: dict[tuple[str, str, int], Any] = {}
    return [PooledModel([known[INCUMBENT], known[partner]], method, cache=cache)
            for partner in POOL_PARTNERS for method in POOL_METHODS]


def log_cpcv(log_path: Path | str, result: CPCVResult, models: Sequence[Any],
             data: pd.DataFrame) -> list[str]:
    """One log row per model of a CPCV run, plus its per-(split, race) losses."""
    params = {m.name: m.params() for m in models}
    plan = result.plan
    seasons = " ".join(str(y) for y in range(CPCV_FIRST_YEAR, CPCV_LAST_YEAR + 1))
    data_id = moment_data_hash(data, result.moment)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows, ids = [], []
    for name in result.models():
        config = params.get(name, {})
        meta = experiment_meta(name, result.moment, config, "cpcv", seasons, data_id,
                               embargo=plan.embargo, blocks=plan.n_blocks,
                               test_blocks=plan.n_test_blocks)
        summary = result.summary(name)
        row: dict[str, Any] = {"timestamp": stamp, "git": git_hash(), "model": name,
                               "params": json.dumps(config, sort_keys=True, default=str),
                               "moment": result.moment, "seasons": seasons,
                               "retrain_every": ""}
        row.update({n: summary[n] for n in METRIC_NAMES})
        row.update(meta)
        rows.append(row)
        write_race_losses(losses_dir_for(log_path), meta["experiment_id"],
                          result.race_losses(name))
        ids.append(meta["experiment_id"])
    append_rows(log_path, rows, LOG_COLUMNS)
    return ids


def write_times(path: Path | str, rows: list[dict[str, Any]]) -> None:
    """Run times of this validation (overwritten every run)."""
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round({"seconds": 1}).to_csv(target, index=False)


# ---------- why B1 / M1 lose to the grid on the winner ----------
def stage_temperatures(scores: np.ndarray, races: np.ndarray, finish: np.ndarray,
                       groups: Sequence[tuple[int, int]] = STAGE_GROUPS) -> pd.DataFrame:
    """The Plackett-Luce temperature beta that fits each group of finishing places best.

    Plackett-Luce picks place k from the drivers still left with probability
    exp(beta * s) / sum(exp(beta * s_left)). One beta serves every place. If the best
    beta for place 1 is much larger than for the midfield, the model spreads too much
    win probability away from the front.
    """
    tensor, mask = finish_tensor(scores, races, finish)
    values = tensor[..., 0]
    rows = []
    for first, last in groups:
        def loss(beta: float, first: int = first, last: int = last) -> float:
            total = 0.0
            for r in range(len(values)):
                n = int(mask[r].sum())
                u = beta * values[r, :n]
                for k in range(first - 1, min(last, n)):
                    total -= u[k] - logsumexp(u[k:])
            return total

        if not any(int(m.sum()) >= first for m in mask):
            continue
        best = optimize.minimize_scalar(loss, bounds=(0.01, 20.0), method="bounded")
        rows.append({"Places": f"{first}" if first == last else f"{first}-{last}",
                     "beta": float(best.x)})
    return pd.DataFrame(rows)


def temperature_study(data: pd.DataFrame, moment: str = POOL_MOMENT,
                      train_until: int = 2023, test: Sequence[int] = WF_SEASONS) -> pd.DataFrame:
    """Out-of-sample stage temperatures of B1 and M1 scores (fit on <= train_until once,
    score the test seasons). beta = 1 means "the fitted scale is right" for those places."""
    from .models import LogisticModel, StrengthModel, as_score_source
    rows = data[(data["Moment"] == moment) & (data["Year"] <= max(test))]
    train = rows[rows["Year"] <= train_until]
    tests = rows[rows["Year"].isin(list(test))]
    if train.empty or tests.empty:
        raise DataNotAvailableError("Not enough rows for the temperature study.")
    out = []
    for model in (LogisticModel(), StrengthModel()):
        source = as_score_source(model.fit(train))
        parts = []
        for _, race in tests.groupby("RaceId", sort=False):
            score, _ = source.scores(strip_outcomes(race))
            parts.append(pd.DataFrame({"RaceId": race["RaceId"].to_numpy(),
                                       "Finish": race["Finish"].to_numpy(float),
                                       "score": np.asarray(score, float)}))
        table = pd.concat(parts, ignore_index=True)
        betas = stage_temperatures(table["score"].to_numpy(), table["RaceId"].to_numpy(),
                                   table["Finish"].to_numpy())
        betas.insert(0, "Model", model.name)
        out.append(betas)
    return pd.concat(out, ignore_index=True)


def front_of_grid_stats(preds: pd.DataFrame, moment: str = POOL_MOMENT,
                        season: int = WF_SEASONS[-1]) -> pd.DataFrame:
    """Per model: mean P_win of the pole-sitter vs how often pole won; mean P_win of the
    favourite vs how often the favourite won; mean P_win given to the actual winner."""
    rows = preds[(preds["Moment"] == moment) & (preds["Year"] == season)]
    out = []
    for model, group in rows.groupby("Model", sort=False):
        pole, fav, winner = [], [], []
        for _, race in group.groupby("RaceId"):
            race = race.reset_index(drop=True)
            if "grid" in race:
                pole_row = race.loc[race["grid"].idxmin()]
            else:
                pole_row = race.loc[race["ExpectedPosition"].idxmin()]
            pole.append((pole_row["P_win"], pole_row["Win"]))
            top = race.loc[race["P_win"].idxmax()]
            fav.append((top["P_win"], top["Win"]))
            won = race[race["Win"] == 1]
            if not won.empty:
                winner.append(float(won["P_win"].iloc[0]))
        p, f = np.array(pole, float), np.array(fav, float)
        out.append({"Model": model, "Pole P_win": p[:, 0].mean(), "Pole won": p[:, 1].mean(),
                    "Favourite P_win": f[:, 0].mean(), "Favourite won": f[:, 1].mean(),
                    "P_win of winner": float(np.mean(winner)), "Races": len(p)})
    return pd.DataFrame(out)


FRONT_FILE, TEMPERATURE_FILE = "validation_front_of_grid.csv", "validation_temperatures.csv"


def save_front_of_grid(preds: pd.DataFrame, data: pd.DataFrame, folder: Path) -> None:
    """Save the evidence for 'Plackett-Luce spreads win probability away from the front'
    (small CSVs next to the log, so --report-only can show it without retraining)."""
    grid = data.loc[data["Moment"] == POOL_MOMENT, ["RaceId", "DriverId", "grid"]]
    joined = preds.merge(grid, on=["RaceId", "DriverId"], how="left")
    singles = joined[joined["Model"].isin([INCUMBENT, *POOL_PARTNERS])]
    folder.mkdir(parents=True, exist_ok=True)
    front_of_grid_stats(singles).to_csv(folder / FRONT_FILE, index=False)
    try:
        temperature_study(data).to_csv(folder / TEMPERATURE_FILE, index=False)
    except (DataNotAvailableError, ValueError):
        (folder / TEMPERATURE_FILE).unlink(missing_ok=True)


def front_of_grid_section(folder: Path) -> list[str]:
    """Markdown from the saved CSVs (nothing if they were never written)."""
    front = folder / FRONT_FILE
    if not front.exists():
        return []
    stats = pd.read_csv(front)
    lines = ["## Why B1 / M1 lose to the grid on the winner (after qualifying)", "",
             f"Walk-forward predictions, {WF_SEASONS[-1]}. 'Pole won' and 'Favourite won' "
             "are what happened; the P_win columns are what each model said on average.",
             "", _table(stats, 3), ""]
    temps = folder / TEMPERATURE_FILE
    if temps.exists():
        betas = pd.read_csv(temps)
        wide = betas.pivot(index="Model", columns="Places", values="beta")
        order = [g for g in ("1", "2-3", "4-10", "11-30") if g in wide.columns]
        wide = wide[order].reset_index()
        lines += ["Best Plackett-Luce temperature per group of places (scores fitted once "
                  "on 2018-2023, measured on 2024-2025; beta = 1: the model's own scale is "
                  "right for those places, beta > 1: the model is too flat there):", "",
                  _table(wide, 2), ""]
    return lines


# ---------- the report ----------
def write_validation_report(log_path: Path | str = EXPERIMENTS_PATH,
                            report_path: Path | str = VALIDATION_REPORT,
                            embargo: int = DEFAULT_EMBARGO, extra: Sequence[str] = (),
                            times_path: Path | str | None = TIMES_PATH,
                            plots: bool = True, note: str = "") -> Path:
    """reports/validation.md from the experiment log only (no training). `note` is
    printed under the title (e.g. which data sources the run had)."""
    log = read_log(log_path)
    current = log[is_validated_phase(log["phase"])]
    if current.empty:
        raise DataNotAvailableError(f"No Phase 15+ experiments in {log_path}: run validate.")
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = _header(current, embargo)
    if note:
        lines[2:2] = [f"> {note}", ""]
    moments = [m for m in ("pre", "practice", "quali") if m in set(current["moment"])]
    for moment in moments:
        lines += moment_section(log, log_path, moment, embargo, target.parent if plots
                                else None)
    lines += embargo_section(current, log_path, target.parent if plots else None)
    decisions = gate_decisions(log, log_path, moments, embargo)
    lines += gates_section(decisions)
    lines += multiple_testing_section(log, log_path, decisions)
    lines += front_of_grid_section(Path(log_path).parent)
    lines += list(extra)
    lines += recommendation_section(decisions)
    lines += times_section(times_path)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _header(current: pd.DataFrame, embargo: int) -> list[str]:
    data_ids = sorted(set(current["data_hash"]) - {""})
    return [
        "# pitwall validation (Phase 15)", "",
        f"Generated {dt.date.today().isoformat()} (git {git_hash()}). Data fingerprint(s): "
        f"{', '.join(data_ids) or 'unknown'}. 2026 is never used.", "",
        "- **Walk-forward** (primary, time-honest): every race of 2024-2025 predicted by "
        "models trained only on earlier races. The gate season is 2025.",
        f"- **CPCV** (robustness check): races 2018-2025 in 8 time blocks, every pair of "
        f"blocks as test (28 splits), {embargo}-race embargo after each test block, "
        "assembled into 7 full backtest paths. Log loss = binary log loss over driver rows.",
        "- Lower log loss is better. P5 / P95 are interpolated from the 7 path scores.", ""]


def _evidence_rows(log: pd.DataFrame, moment: str, embargo: int) -> tuple[list[str], str]:
    rows = log[(log["moment"] == moment) & is_validated_phase(log["phase"])]
    cpcv = rows[(rows["validation"] == "cpcv") & (rows["embargo"] == str(embargo))]
    data_id = cpcv["data_hash"].iloc[-1] if not cpcv.empty else (
        rows["data_hash"].iloc[-1] if not rows.empty else "")
    names = list(dict.fromkeys(rows.loc[rows["data_hash"] == data_id, "model"]))
    return names, data_id


def moment_section(log: pd.DataFrame, log_path: Path | str, moment: str, embargo: int,
                   plot_dir: Path | None) -> list[str]:
    """Walk-forward vs CPCV path distribution per model at one moment."""
    names, data_id = _evidence_rows(log, moment, embargo)
    table, paths, wf_win = [], [], {}
    for name in [n for n in names if "(l2=" not in n]:
        try:
            ev = load_evidence(name, moment, log_path, embargo=embargo, data=data_id)
        except DataNotAvailableError:
            continue
        cells = [name, f"{weighted_loss(ev.walk_forward, 'win'):.4f}",
                 f"{weighted_loss(ev.season(2025), 'win'):.4f}"
                 if not ev.season(2025).empty else "—"]
        wf_win[name] = weighted_loss(ev.walk_forward, "win")
        for kind in ("win", "podium"):
            if ev.cpcv is None:
                cells += ["—"] * 4
                continue
            scores = path_losses(ev.cpcv, ev.cpcv_blocks, ev.cpcv_test_blocks, kind)
            d = distribution(scores)
            cells += [f"{d['Mean']:.4f}", f"{d['P5']:.4f}", f"{d['Median']:.4f}",
                      f"{d['P95']:.4f}"]
            if kind == "win":
                paths += [{"Model": name, "Path": i, "LogLoss": s} for i, s in enumerate(scores)]
        table.append(cells)
    header = ["Model", "WF win LL 2024-25", "WF win LL 2025",
              "CPCV win mean", "win P5", "win median", "win P95",
              "CPCV podium mean", "podium P5", "podium median", "podium P95"]
    lines = [f"## Moment: {moment}", ""]
    if not table:
        return lines + ["No experiments logged for this moment.", ""]
    lines += [markdown_table(header, table), ""]
    if plot_dir is not None and paths:
        image = _save_boxplot(pd.DataFrame(paths), pd.Series(wf_win), moment, plot_dir)
        lines += [f"![CPCV paths, {moment}]({image.name})", ""]
    return lines


def _save_boxplot(paths: pd.DataFrame, wf: pd.Series, moment: str, folder: Path) -> Path:
    import matplotlib.pyplot as plt

    from ..plotting import cpcv_boxplot
    models = paths["Model"].nunique()
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * models + 2), 4.5))
    cpcv_boxplot(paths, wf, title=f"CPCV win log loss per path, moment '{moment}' "
                                  "(diamond = walk-forward 2024-25)", ax=ax)
    target = folder / f"validation_cpcv_{moment}.png"
    fig.tight_layout()
    fig.savefig(target, dpi=110)
    plt.close(fig)
    return target


def embargo_table(current: pd.DataFrame, log_path: Path | str) -> pd.DataFrame:
    """Mean CPCV win log loss per (moment, model, embargo) on the latest data."""
    rows = []
    cpcv = current[current["validation"] == "cpcv"]
    for (moment, model, embargo), group in cpcv.groupby(["moment", "model", "embargo"]):
        latest = group.iloc[-1]
        losses = read_race_losses(losses_dir_for(log_path), latest["experiment_id"])
        scores = path_losses(losses, int(latest["cpcv_blocks"]),
                             int(latest["cpcv_test_blocks"]), "win")
        rows.append({"Moment": moment, "Model": model, "Embargo": int(embargo),
                     "LogLoss": float(scores.mean())})
    return pd.DataFrame(rows, columns=["Moment", "Model", "Embargo", "LogLoss"])


def embargo_section(current: pd.DataFrame, log_path: Path | str,
                    plot_dir: Path | None) -> list[str]:
    """How much the CPCV scores move with the embargo, and the recommended default."""
    table = embargo_table(current, log_path)
    counts = table.groupby(["Moment", "Model"])["Embargo"].nunique()
    keep = counts[counts > 1].index
    table = table.set_index(["Moment", "Model"]).loc[keep].reset_index() if len(keep) else \
        table.iloc[0:0]
    lines = ["## Embargo sensitivity (CPCV mean win log loss)", ""]
    if table.empty:
        return lines + ["Only one embargo was run.", ""]
    wide = table.pivot_table(index=["Moment", "Model"], columns="Embargo",
                             values="LogLoss").reset_index()
    embargoes = [int(c) for c in wide.columns if isinstance(c, (int, np.integer))]
    wide["Spread"] = wide[embargoes].max(axis=1) - wide[embargoes].min(axis=1)
    rows = [[str(r["Moment"]), str(r["Model"]),
             *[f"{r[e]:.4f}" if not math.isnan(r[e]) else "—" for e in embargoes],
             f"{r['Spread']:.4f}"] for r in wide.to_dict("records")]
    header = ["Moment", "Model", *[f"embargo {e}" for e in embargoes], "max - min"]
    lines += [markdown_table(header, rows), ""]
    lines += embargo_recommendation(wide, embargoes)
    if plot_dir is not None:
        import matplotlib.pyplot as plt

        from ..plotting import embargo_plot
        fig, ax = plt.subplots(figsize=(9, 4.5))
        embargo_plot(table, ax=ax)
        target = plot_dir / "validation_embargo.png"
        fig.tight_layout()
        fig.savefig(target, dpi=110)
        plt.close(fig)
        lines += [f"![Embargo sensitivity]({target.name})", ""]
    return lines


def embargo_recommendation(wide: pd.DataFrame, embargoes: list[int]) -> list[str]:
    """Plain-words recommendation from the sensitivity table."""
    worst = wide.loc[wide["Spread"].idxmax()]
    low, high = min(embargoes), max(embargoes)
    direction = (wide[low] - wide[high]).dropna() if low in wide and high in wide else None
    lines = [f"Largest move: {worst['Model']} at '{worst['Moment']}', "
             f"{worst['Spread']:.4f} win LL (tolerance {EMBARGO_TOLERANCE})."]
    if direction is not None and len(direction):
        better = int((direction < 0).sum())
        lines.append(f"With embargo {low} the score is lower (looks better) than with "
                     f"embargo {high} for {better} of {len(direction)} model-moments: "
                     "lower-with-less-embargo is the signature of leakage.")
    if float(wide["Spread"].max()) > EMBARGO_TOLERANCE:
        lines.append(f"**Recommendation:** the scores move by more than {EMBARGO_TOLERANCE}"
                     ", so use a larger default embargo (10 race weekends covers the 10-race "
                     "rolling windows; championship points still carry a whole season, so "
                     "no finite embargo removes everything).")
    else:
        lines.append(f"**Recommendation:** the scores move by less than {EMBARGO_TOLERANCE}"
                     " for every model: keep the default embargo of 1 race weekend (larger "
                     "embargoes only remove training races).")
    return lines + [""]


def gate_decisions(log: pd.DataFrame, log_path: Path | str, moments: Sequence[str],
                   embargo: int) -> list[GateDecision]:
    """gate_v2 for every logged pool (quali) and a re-check of every other single model,
    each against B0 at the same moment, from logged losses only."""
    decisions = []
    for moment in moments:
        names, data_id = _evidence_rows(log, moment, embargo)
        try:
            incumbent = load_evidence(INCUMBENT, moment, log_path, embargo=embargo,
                                      data=data_id)
        except DataNotAvailableError:
            continue
        pools = [n for n in names if n.startswith("POOL-") and "(l2=" not in n]
        singles = [n for n in names if not n.startswith("POOL-") and n != INCUMBENT]
        for name in [*pools, *singles]:
            try:
                cand = load_evidence(name, moment, log_path, embargo=embargo, data=data_id)
            except DataNotAvailableError:
                continue
            trials = max(count_trials(log, model_family(name), moment), 1)
            decisions.append(gate_v2(cand, incumbent, n_trials=trials))
    return decisions


def gates_section(decisions: Sequence[GateDecision]) -> list[str]:
    """Every gate decision with its reasons."""
    lines = ["## Gate decisions (gate_v2, against B0 grid slot)", "",
             "Required: the 95% paired-bootstrap CI of (candidate - B0) on 2025 entirely "
             "below 0 for win AND podium log loss; the deflated p < 0.05 once the family has "
             ">= 10 trials; the median CPCV path better than B0's. Diebold-Mariano and the "
             "3-race block bootstrap are context.", ""]
    if not decisions:
        return lines + ["No gate could be run (no matching experiments).", ""]
    summary = [[d.moment, d.candidate, str(d.n_trials), "PASS" if d.passed else "fail"]
               for d in decisions]
    lines += [markdown_table(["Moment", "Candidate", "Trials", "Gate"], summary), ""]
    for d in decisions:
        lines += d.to_markdown()
    return lines


def multiple_testing_section(log: pd.DataFrame, log_path: Path | str,
                             decisions: Sequence[GateDecision]) -> list[str]:
    """Trials per family, PBO per family with >= 2 configurations, BH over the gates."""
    lines = ["## Multiple testing", "", "### Trials per family (distinct configurations, "
             "from reports/experiments.csv)", ""]
    trials = trials_table(log)
    lines += [_table(trials, 0), ""]
    lines += ["### Probability of Backtest Overfitting", ""]
    pbo_rows = pbo_by_family(log, log_path)
    if pbo_rows:
        lines += [markdown_table(["Family", "Moment", "Configs", "Races", "PBO", "Reading"],
                                 pbo_rows), ""]
    else:
        lines += ["No family has two or more configurations with walk-forward losses on the "
                  "same races yet.", ""]
    if decisions:
        bh = benjamini_hochberg([d.p_value for d in decisions])
        rows = [[d.moment, d.candidate, f"{d.p_value:.4f}", f"{a:.4f}",
                 "discovery" if r else "no"]
                for d, a, r in zip(decisions, bh["adjusted"], bh["rejected"])]
        lines += ["### Benjamini-Hochberg over every gate of this phase (q = 0.05)", "",
                  "p = one-sided paired-bootstrap p-value (win LL, 2025) that the candidate "
                  "is better than B0.", "",
                  markdown_table(["Moment", "Candidate", "p", "BH-adjusted", "Result"], rows),
                  ""]
    return lines


def pbo_by_family(log: pd.DataFrame, log_path: Path | str) -> list[list[str]]:
    """PBO over all walk-forward configurations of each family (common races only)."""
    out = []
    wf = log[(log["validation"] == "walk_forward") & is_validated_phase(log["phase"])]
    for (family, moment), group in wf.groupby(["family", "moment"]):
        latest = group.groupby("params_hash").tail(1)
        if len(latest) < 2:
            continue
        columns = {}
        for row in latest.to_dict("records"):
            try:
                losses = read_race_losses(losses_dir_for(log_path), row["experiment_id"])
            except FileNotFoundError:
                continue
            columns[row["model"]] = losses.set_index("RaceId")["win"]
        if len(columns) < 2:
            continue
        table = pd.concat(columns, axis=1, join="inner").sort_index()
        try:
            res = probability_of_backtest_overfitting(table)
        except ValueError as exc:
            out.append([family, moment, str(table.shape[1]), str(table.shape[0]), "—",
                        f"not computed: {exc}"])
            continue
        reading = ("looks like luck" if res.pbo >= 0.3 else "best config holds up")
        out.append([family, moment, str(res.n_configs), str(res.n_races), f"{res.pbo:.2f}",
                    reading])
    return out


def recommendation_section(decisions: Sequence[GateDecision]) -> list[str]:
    """What to do with reports/final_config.json (NOT rewritten in this phase)."""
    pools = [d for d in decisions if d.candidate.startswith("POOL-")]
    lines = ["## Recommendation for reports/final_config.json (not changed in Phase 15)", ""]
    passed = [d for d in pools if d.passed]
    if not pools:
        lines.append("No pool was gated.")
    elif passed:
        names = ", ".join(d.candidate for d in passed)
        lines.append(f"Passed the gate after qualifying: {names}. Recommendation: switch the "
                     "quali moment to the best of these in a later phase, after re-running "
                     "the gate on the full dataset.")
    else:
        lines.append("No B0 + B1 / B0 + M1 pool passes gate_v2 after qualifying. "
                     "Recommendation: keep the plain grid baseline (B0) at the quali moment, "
                     "as final_config.json already does.")
    singles = [d for d in decisions if not d.candidate.startswith("POOL-") and d.passed]
    for d in singles:
        lines.append(f"Re-check: {d.candidate} passes gate_v2 against B0 at '{d.moment}'; "
                     "worth a dedicated phase before any config change.")
    return lines + [""]


def times_section(path: Path | str | None) -> list[str]:
    """Run times of the last validation (if recorded)."""
    if path is None or not Path(path).exists():
        return []
    times = pd.read_csv(path)
    total = times.groupby(["validation", "model"])["seconds"].sum().reset_index()
    return ["## Run times (last validation run, this machine)", "", _table(total, 0),
            f"Total: {times['seconds'].sum() / 60:.1f} min.", ""]


def _table(frame: pd.DataFrame, decimals: int) -> str:
    rows = []
    for row in frame.to_dict("records"):
        cells = []
        for value in row.values():
            if isinstance(value, float):
                cells.append(f"{value:.{decimals}f}" if decimals else f"{value:.0f}")
            else:
                cells.append(str(value))
        rows.append(cells)
    return markdown_table([str(c) for c in frame.columns], rows)


def evidence_for(model: str, moment: str, log_path: Path | str = EXPERIMENTS_PATH,
                 embargo: int | None = DEFAULT_EMBARGO) -> Evidence:
    """Shortcut used by `pitwall-f1 gate`."""
    return load_evidence(model, moment, log_path, embargo=embargo)
