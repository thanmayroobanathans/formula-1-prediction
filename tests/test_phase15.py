"""Phase 15: CPCV, statistics, experiment log v2, gate_v2, CLI. Offline, synthetic data.

Every statistic is checked on a case where the true answer is known: two identical
models must NOT pass; a planted real improvement MUST pass with enough races; PBO is
near 0.5 for pure-noise configurations.
"""
import json
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from synthetic_f1 import WORLD_PARAMS, make_dataset, make_strength_world

from pitwall import cli
from pitwall.predict import GridSlotBaseline, LogisticModel, StrengthModel, walk_forward
from pitwall.predict.backtest import (
    LOG_COLUMNS, LOG_COLUMNS_V1, Evidence, LeakageError, LockedSeasonError, gate,
    gate_v2, load_evidence, run_backtest,
)
from pitwall.predict.ensemble import PooledModel, pool_forward, pool_name, pool_params
from pitwall.predict.experiments import (
    append_rows, config_params, count_trials, data_hash, find_experiment, losses_dir_for,
    merge_logs, model_family, params_hash, read_log, read_race_losses, trials_table,
    write_race_losses, PHASE,
)
from pitwall.predict.metrics import race_loss_table, summarise, weighted_loss
from pitwall.predict.models.base import check_consistency
from pitwall.predict.significance import (
    benjamini_hochberg, block_bootstrap, bootstrap_p_value, deflated_score, diebold_mariano,
    expected_max_sharpe, paired_bootstrap, probabilistic_sharpe,
    probability_of_backtest_overfitting,
)
from pitwall.predict.strength import StrengthFilter, observation_table
from pitwall.predict.strength_features import StrengthFeatures
from pitwall.predict.validation import (
    CPCVSplit, assemble_paths, check_split, cpcv_race_losses, distribution, n_paths,
    path_losses, path_race_losses, plan_cpcv, race_calendar, run_cpcv, split_blocks,
)
from pitwall.predict.validation_report import (
    LoggedConfig, ValidateOptions, embargo_recommendation, front_of_grid_stats,
    log_walk_forward_pools, model_factories_for, run_validation, save_front_of_grid,
    stage_temperatures, write_validation_report,
)

FAST = 2000
YEARS = tuple(range(2018, 2027))  # 2026 is in the data, and must never be used


@pytest.fixture(scope="module")
def data():
    return make_dataset(seasons=YEARS, races_per_season=4, n_drivers=8, seed=3)


@pytest.fixture(scope="module")
def plan(data):
    return plan_cpcv(data)


# ---------- CPCV: splits, embargo, paths ----------
def test_split_and_path_counts():
    assert len(split_blocks(8, 2)) == math.comb(8, 2) == 28
    assert n_paths(8, 2) == 7 and n_paths(6, 2) == 5 and n_paths(10, 3) == 36
    paths = assemble_paths(8, 2)
    assert len(paths) == 7
    tests = split_blocks(8, 2)
    used = set()
    for path in paths:
        assert sorted(path) == list(range(8))               # every block exactly once
        for block, split in path.items():
            assert block in tests[split]                   # that split really tests it
            used.add((block, split))
    assert len(used) == 8 * 7                              # every (block, split) once
    with pytest.raises(ValueError):
        split_blocks(4, 4)


def test_plan_blocks_never_touch_2026(data, plan):
    assert plan.calendar["Year"].max() == 2025 and 2026 not in set(plan.calendar["Year"])
    assert len(plan.calendar) == 32 and plan.calendar["Block"].value_counts().eq(4).all()
    assert plan.calendar["Date"].is_monotonic_increasing
    with pytest.raises(LockedSeasonError):
        plan_cpcv(data, last_year=2026)
    with pytest.raises(LockedSeasonError):
        race_calendar(data, last_year=2026)


@pytest.mark.parametrize("embargo", [0, 1, 3])
def test_splits_have_no_overlap_and_respect_the_embargo(data, embargo):
    plan = plan_cpcv(data, embargo=embargo)
    position = dict(zip(plan.calendar["RaceId"], plan.calendar["Position"]))
    for split in plan.splits:
        train, test = set(split.train), set(split.test)
        assert not train & test and not train & set(split.embargoed)
        assert train | test | set(split.embargoed) | set(split.purged) == set(position)
        test_pos = np.array([position[r] for r in test])
        for race in train:
            gap = position[race] - test_pos
            assert not ((gap > 0) & (gap <= embargo)).any()
        assert not split.purged  # one-race labels: the purge removes nothing
        check_split(split, plan.calendar, embargo)
        # embargoed = the `embargo` races after each test run (cut at the calendar end)
        assert len(split.embargoed) <= embargo * len(split.test_blocks)
    assert sum(len(s.embargoed) for s in plan.splits) > 0 or embargo == 0


def test_check_split_catches_leaks(plan):
    split = plan.splits[0]  # test blocks (0, 1): races 0-7, embargo 1 -> race 8 removed
    ids = plan.calendar["RaceId"].tolist()
    leaky = CPCVSplit(split.index, split.test_blocks, split.train + (ids[8],), split.test,
                      (), ())
    with pytest.raises(LeakageError, match="embargo"):
        check_split(leaky, plan.calendar, embargo=1)
    inside = CPCVSplit(split.index, split.test_blocks, split.train + (ids[0],), split.test,
                       (), ())
    with pytest.raises(LeakageError, match="both"):
        check_split(inside, plan.calendar, embargo=1)


def test_run_cpcv_predicts_every_race_once_per_path(data, plan):
    res = run_cpcv([GridSlotBaseline()], data, "quali", plan)
    preds = res.predictions
    assert set(preds["Split"]) == set(range(28))
    per_race = preds.drop_duplicates(["Split", "RaceId"]).groupby("RaceId").size()
    assert (per_race == 7).all() and len(per_race) == 32    # C(7, 1) splits per block
    check_consistency(preds[preds["Split"] == 0])
    losses = res.race_losses("B0 grid slot")
    assert set(losses.columns) >= {"Split", "Block", "RaceId", "n_rows", "win", "winner"}
    scores = path_losses(losses, 8, 2, "win")
    assert len(scores) == 7 and np.all(scores > 0)
    for p in range(7):
        assert path_race_losses(losses, 8, 2, p)["RaceId"].nunique() == 32
    assert res.path_scores()["Path"].nunique() == 7
    d = distribution(scores)
    assert d["P5"] <= d["Median"] <= d["P95"] and d["Paths"] == 7
    assert summarise(preds)["n_races"] == 32
    with pytest.raises(ValueError):
        path_losses(losses[losses["Block"] != 0], 8, 2)


def test_training_rows_never_include_a_test_race(data, plan):
    seen = []

    class Spy(GridSlotBaseline):
        name = "spy"

        def fit(self, train):
            seen.append(set(train["RaceId"]))
            return super().fit(train)

    run_cpcv([Spy()], data, "pre", plan)
    for split, train in zip(plan.splits, seen):
        assert train == set(split.train)


# ---------- fix: M1 forecasts a PAST race from the filter state before it ----------
def test_strength_features_forecast_past_races_from_the_past_only():
    world, _ = make_strength_world()
    rows = world[world["Moment"] == "quali"]
    early = rows[rows["RaceId"] == 15]
    features = StrengthFeatures(WORLD_PARAMS)
    features.fit(rows[rows["RaceId"] != 15])           # trained THROUGH later races
    got = features.for_race(early.drop(columns=["Finish"]))
    past = observation_table(rows[rows["Date"] < early["Date"].iloc[0]])
    want = StrengthFilter(WORLD_PARAMS).run(past, record=False).forecast(early)
    np.testing.assert_allclose(got["pace_pred"], want["pace_pred"].to_numpy())
    later = rows[rows["RaceId"] == rows["RaceId"].max()]
    walk = StrengthFeatures(WORLD_PARAMS)
    walk.fit(rows[rows["RaceId"] < later["RaceId"].iloc[0]])   # walk-forward: unchanged
    np.testing.assert_allclose(walk.for_race(later)["pace_pred"],
                               walk.filter_.forecast(later)["pace_pred"].to_numpy())
    assert walk.filter_before(later["Date"].iloc[0]) is walk.filter_


# ---------- pools ----------
def test_pooled_model_uses_training_races_only(data):
    quali = data[(data["Moment"] == "quali") & (data["Year"] <= 2023)]
    seen = []

    class SpyB1(LogisticModel):
        def fit(self, train):
            seen.append(set(train["RaceId"]))
            return super().fit(train)

    pooled = PooledModel([GridSlotBaseline, lambda: SpyB1(n_sims=FAST)], "log",
                         cache={})
    assert pooled.name == "POOL-log B0+B1" == pool_name(["B0 grid slot", "B1 logistic"],
                                                        "log")
    assert pooled.params() == pool_params(["B0 grid slot", "B1 logistic"], "log", 0.05)
    pooled.fit(quali)
    assert all(s <= set(quali["RaceId"]) for s in seen) and len(seen) == 4
    assert math.isclose(pooled.weights_.sum(), 1.0) and (pooled.weights_ > 0).all()
    race = data[(data["Moment"] == "quali") & (data["RaceId"] == 30)]
    check_consistency(pooled.predict(race))
    again = PooledModel([GridSlotBaseline, lambda: SpyB1(n_sims=FAST)], "linear",
                        cache=pooled.cache)
    count = len(seen)
    again.fit(quali)
    assert len(seen) == count  # members reused from the shared cache
    with pytest.raises(ValueError):
        PooledModel([GridSlotBaseline], "median")
    with pytest.raises(ValueError):
        PooledModel([GridSlotBaseline], "log").fit(quali[quali["RaceId"] <= 3])


def test_pool_forward_names_and_respects_the_past(data):
    models = [GridSlotBaseline(), LogisticModel(n_sims=FAST)]
    preds = walk_forward(models, data, "quali", seasons=[2024, 2025])
    table, history = pool_forward(preds, "quali", ["B0 grid slot", "B1 logistic"], "log",
                                  [2025], [2024], l2=0.2, min_train_races=2)
    assert set(table["Model"]) == {"POOL-log B0+B1 (l2=0.2)"}
    assert (history["TrainRaces"] >= 2).all()
    check_consistency(table.drop(columns="Eval"))


# ---------- statistics on known cases ----------
def _pair(n=48, gap=0.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Series(0.12 + rng.normal(0, 0.02, n), index=np.arange(1, n + 1))
    better = base - gap + pd.Series(rng.normal(0, noise, n), index=base.index)
    return better, base


def test_identical_models_never_pass():
    a, b = _pair()
    same = a.copy()
    dm = diebold_mariano(same - same)
    assert dm.statistic == 0 and dm.p_value == 1.0 and dm.p_better == pytest.approx(0.5)
    assert not paired_bootstrap(same, same).significant
    assert not block_bootstrap(same, same).high < 0
    assert bootstrap_p_value(same, same) > 0.5
    assert deflated_score((same - same).to_numpy(), n_trials=1).p_value == 1.0
    assert not benjamini_hochberg([1.0, 0.9])["rejected"].any()


def test_planted_improvement_passes_with_enough_races():
    a, b = _pair(gap=0.01, noise=0.01)
    assert paired_bootstrap(a, b).high < 0
    assert block_bootstrap(a, b, block=3).high < 0
    assert diebold_mariano(a - b).p_better < 0.01
    assert bootstrap_p_value(a, b) < 0.01
    assert deflated_score((b - a).to_numpy(), n_trials=10).p_value < 0.05
    few_a, few_b = _pair(n=6, gap=0.002, noise=0.01, seed=4)   # too few races: not shown
    assert not paired_bootstrap(few_a, few_b).high < 0


def test_diebold_mariano_by_hand():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    res = diebold_mariano(d)
    dm = 2.5 / math.sqrt(1.25 / 4)
    assert res.statistic == pytest.approx(dm * math.sqrt(3 / 4))  # HLN: (n + 1 - 2) / n
    assert res.p_value == pytest.approx(2 * stats.t.sf(res.statistic, 3))
    lag2 = diebold_mariano(d, lag=2)
    assert lag2.lag == 2 and lag2.n == 4
    with pytest.raises(ValueError):
        diebold_mariano([1.0])
    with pytest.raises(ValueError):
        diebold_mariano(d, lag=4)
    assert math.isinf(diebold_mariano(np.full(5, -0.1)).statistic)


def test_block_bootstrap_is_wider_for_dependent_races():
    rng = np.random.default_rng(1)
    runs = np.repeat(rng.normal(0, 0.02, 12), 4)        # 4 identical neighbours in a row
    a = pd.Series(runs, index=np.arange(48))
    b = pd.Series(np.zeros(48), index=a.index)
    narrow = block_bootstrap(a, b, block=1)
    wide = block_bootstrap(a, b, block=4)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)
    with pytest.raises(ValueError):
        block_bootstrap(a, b, block=0)
    with pytest.raises(ValueError):
        block_bootstrap(a.iloc[:2].set_axis([100, 101]), b.iloc[:2])
    dates = pd.Series(pd.date_range("2025-01-01", periods=48)[::-1], index=a.index)
    assert block_bootstrap(a, b, order=dates).mean == pytest.approx(narrow.mean)


def test_expected_maximum_and_deflation():
    assert expected_max_sharpe(1, 1.0) == 0.0
    z = stats.norm.ppf
    g = 0.5772156649015329
    assert expected_max_sharpe(30, 1.0) == pytest.approx((1 - g) * z(1 - 1 / 30)
                                                         + g * z(1 - 1 / (30 * math.e)))
    values = [expected_max_sharpe(n, 0.2) for n in (2, 10, 30, 100)]
    assert values == sorted(values)                     # more tries, more luck
    a, b = _pair(gap=0.004, noise=0.012, seed=2)
    p = [deflated_score((b - a).to_numpy(), n).p_value for n in (1, 10, 100)]
    assert p == sorted(p)
    res = deflated_score((b - a).to_numpy(), 10)
    assert "useless tries" in res.explain() and res.n_trials == 10
    assert probabilistic_sharpe(0.0, 40, 0.0, 3.0) == pytest.approx(0.5)
    assert math.isnan(probabilistic_sharpe(0.1, 1, 0.0, 3.0))
    assert deflated_score(np.full(5, 0.01), 3).p_value == 0.0   # a constant real gain
    with pytest.raises(ValueError):
        deflated_score([0.1, 0.2], 1)


def test_pbo_is_about_one_half_for_noise_and_near_zero_for_a_real_winner():
    noise = [probability_of_backtest_overfitting(
        pd.DataFrame(np.random.default_rng(s).normal(0, 1, (64, 20)))).pbo
        for s in range(12)]
    assert 0.35 < np.mean(noise) < 0.65
    rng = np.random.default_rng(0)
    table = pd.DataFrame(rng.normal(0, 1, (64, 10)))
    table[3] -= 1.0                                   # configuration 3 is truly better
    res = probability_of_backtest_overfitting(table)
    assert res.pbo < 0.1 and res.n_combinations == math.comb(8, 4)
    assert "PBO" in res.explain()
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(table[[0]])
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(table, n_blocks=3)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(table.iloc[:5])


def test_benjamini_hochberg_by_hand():
    res = benjamini_hochberg([0.01, 0.02, 0.03, 0.5])
    assert res["rejected"].tolist() == [True, True, True, False]
    assert res["adjusted"].tolist() == pytest.approx([0.04, 0.04, 0.04, 0.5])
    assert benjamini_hochberg([0.04, 0.03])["rejected"].all()
    assert not benjamini_hochberg([0.03, 0.9, 0.8])["rejected"].any()
    assert benjamini_hochberg([]).empty


# ---------- experiment log v2 ----------
def test_names_hashes_and_trials():
    assert model_family("M1-pure strength") == "M1" and model_family("POOL-log B0+B1") == "pool"
    assert model_family("ENS-linear+cal") == "ensemble" and model_family("odd one") == "odd"
    fitted = {"l2": 1.0, "strength": {"q_car": 0.1}}
    assert config_params(fitted) == {"l2": 1.0}
    assert params_hash(fitted) == params_hash({"l2": 1.0, "strength": {"q_car": 9.9}})
    assert params_hash(json.dumps(fitted)) == params_hash(fitted)
    assert params_hash("not json") != params_hash("")
    assert data_hash(pd.DataFrame()) == "empty"
    log = pd.DataFrame({"family": ["M1", "M1", "M1", "B1"],
                        "moment": ["quali"] * 4,
                        "params_hash": ["a", "a", "b", "c"]})
    assert count_trials(log, "M1", "quali") == 2 and count_trials(log.iloc[0:0], "M1", "q") == 0
    assert trials_table(log).set_index("family").loc["M1", "trials"] == 2
    assert trials_table(pd.DataFrame()).empty


def test_log_round_trip_and_old_rows_stay_readable(data, tmp_path):
    log = tmp_path / "experiments.csv"
    old = {c: "" for c in LOG_COLUMNS_V1}
    old.update({"timestamp": "2026-11-25T02:12:14+00:00", "git": "04e03e2",
                "model": "M1-pure strength", "params": json.dumps({"pure": True}),
                "moment": "quali", "seasons": "2024 2025", "win_logloss": "0.1655"})
    pd.DataFrame([old])[LOG_COLUMNS_V1].to_csv(log, index=False)     # a Phase 14 file
    result = run_backtest([GridSlotBaseline(), LogisticModel(n_sims=FAST)], data,
                          seasons=[2025], moments=["quali"], log_path=log)
    table = read_log(log)
    assert list(pd.read_csv(log, nrows=0).columns) == LOG_COLUMNS
    first = table.iloc[0]
    assert first["family"] == "M1" and first["phase"] == "<=14"
    assert first["validation"] == "walk_forward" and first["win_logloss"] == "0.1655"
    new = table[table["phase"] == PHASE]
    assert len(new) == 2 and new["experiment_id"].is_unique
    assert set(new["validation"]) == {"walk_forward"} and (new["data_hash"] != "").all()
    for row in new.to_dict("records"):
        losses = read_race_losses(losses_dir_for(log), row["experiment_id"])
        assert weighted_loss(losses, "win") == pytest.approx(float(row["win_logloss"]),
                                                             rel=1e-5)
        assert len(losses) == 4
    # the same experiment again: nothing new is logged
    run_backtest([GridSlotBaseline()], data, seasons=[2025], moments=["quali"], log_path=log)
    assert len(read_log(log)) == 3
    row = find_experiment(read_log(log), "B0 grid slot", "quali", "walk_forward",
                          season=2025)
    assert row is not None and find_experiment(table, "nope", "quali", "cpcv") is None
    assert find_experiment(pd.DataFrame(), "B0", "quali", "cpcv") is None
    assert summarise(result.predictions[result.predictions["Model"] == "B0 grid slot"])
    with pytest.raises(FileNotFoundError):
        read_race_losses(losses_dir_for(log), "missing")
    assert read_log(tmp_path / "none.csv").empty


def test_race_loss_table_matches_summarise(data):
    preds = walk_forward([GridSlotBaseline()], data, "pre", seasons=[2025])
    table = race_loss_table(preds)
    assert len(table) == 4 and table["Date"].is_monotonic_increasing
    full = summarise(preds)
    assert weighted_loss(table, "win") == pytest.approx(full["win_logloss"])
    assert table["winner"].mean() == pytest.approx(full["winner_logloss"])


def test_append_rows_writes_new_files_and_merge(tmp_path):
    rows = [{"experiment_id": "x1", "model": "B0 grid slot", "moment": "pre"}]
    target = tmp_path / "a" / "experiments.csv"
    assert append_rows(target, rows, LOG_COLUMNS) == 1
    assert append_rows(target, rows, LOG_COLUMNS) == 0
    write_race_losses(losses_dir_for(target), "x1",
                      pd.DataFrame({"RaceId": [1], "Date": pd.to_datetime(["2025-03-01"]),
                                    "n_rows": [20], "win": [0.1]}))
    (target.parent / "validation_times.csv").write_text("moment,seconds\npre,1\n")
    (target.parent / "validation_front_of_grid.csv").write_text("Model\nB0\n")
    merged = tmp_path / "b" / "experiments.csv"
    assert merge_logs([target, tmp_path / "missing.csv"], merged, LOG_COLUMNS) == 1
    assert (losses_dir_for(merged) / "x1.csv").exists()
    assert (merged.parent / "validation_front_of_grid.csv").exists()
    assert (merged.parent / "validation_times.csv").exists()


# ---------- gate_v2 ----------
def _evidence(name, values, cpcv_shift=None, seed=0):
    n = len(values)
    dates = pd.date_range("2025-03-01", periods=n, freq="14D")
    wf = pd.DataFrame({"RaceId": np.arange(1, n + 1), "Date": dates, "Year": 2025,
                       "n_rows": 20, "win": values, "podium": np.asarray(values) * 2,
                       "points": 0.5, "dnf": 0.3, "winner": 1.5})
    cpcv = None
    if cpcv_shift is not None:
        rng = np.random.default_rng(seed)
        rows = []
        for split, blocks in enumerate(combinations(range(8), 2)):
            for block in blocks:
                for race in range(block * 3, block * 3 + 3):
                    rows.append({"Split": split, "Block": block, "RaceId": race,
                                 "n_rows": 20, "win": 0.12 + cpcv_shift
                                 + rng.normal(0, 0.002), "podium": 0.24})
        cpcv = pd.DataFrame(rows)
    return Evidence(name, "quali", wf, cpcv, family=model_family(name))


def test_gate_v2_decisions():
    a, b = _pair(n=24, gap=0.01, noise=0.01)
    good = _evidence("POOL-log B0+B1", a.to_numpy(), cpcv_shift=-0.01)
    base = _evidence("B0 grid slot", b.to_numpy(), cpcv_shift=0.0)
    decision = gate_v2(good, base, n_trials=1)
    assert decision.passed and "PASSES" in decision.verdict()
    assert any("context" in r for r in decision.reasons())
    assert "| Check |" in "\n".join(decision.to_markdown())
    twin = gate_v2(_evidence("B1 twin", b.to_numpy(), cpcv_shift=0.0), base)
    assert not twin.passed and "does NOT pass" in twin.verdict()
    no_cpcv = gate_v2(_evidence("POOL-x", a.to_numpy()), base)
    assert not no_cpcv.passed
    assert any("no CPCV run logged" in c.detail for c in no_cpcv.checks)
    worse_cpcv = gate_v2(_evidence("POOL-y", a.to_numpy(), cpcv_shift=+0.01), base)
    assert not worse_cpcv.passed
    weak_a, weak_b = _pair(n=24, gap=0.004, noise=0.012, seed=5)
    weak = _evidence("POOL-z", weak_a.to_numpy(), cpcv_shift=-0.01)
    weak_base = _evidence("B0 grid slot", weak_b.to_numpy(), cpcv_shift=0.0)
    one = gate_v2(weak, weak_base, n_trials=1)
    many = gate_v2(weak, weak_base, n_trials=200)
    deflation = [c for c in many.checks if "deflated" in c.name][0]
    assert deflation.required and not [c for c in one.checks if "deflated" in c.name][0].required
    assert many.p_value == one.p_value
    assert not many.passed or deflation.passed
    with pytest.raises(LockedSeasonError):
        gate_v2(good, base, season=2026)
    with pytest.raises(ValueError):
        gate_v2(good, Evidence("x", "pre", base.walk_forward))
    odd = _evidence("POOL-w", a.to_numpy(), cpcv_shift=-0.01)
    shorter = Evidence(odd.model, odd.moment, odd.walk_forward,
                       odd.cpcv[odd.cpcv["Block"] != 7])
    assert any("not comparable" in c.detail for c in gate_v2(shorter, base).checks)


def test_old_gate_still_works(data):
    result = run_backtest([GridSlotBaseline()], data, seasons=[2025], moments=["quali"],
                          log_path=None)
    decision = gate(result, "B0 grid slot", ["B0 grid slot"])
    assert not decision.kept


# ---------- the whole validation, the report and the CLI ----------
@pytest.fixture(scope="module")
def validated(data, tmp_path_factory):
    folder = tmp_path_factory.mktemp("validate")
    log = folder / "experiments.csv"
    options = ValidateOptions(models=["B0 grid slot", "B1 logistic"], moments=["quali"],
                              sensitivity=(0,), pools=False, log_path=log,
                              report_path=folder / "validation.md",
                              times_path=folder / "validation_times.csv")
    lines = run_validation(data, options, progress=lambda text: None)
    return folder, log, lines


def test_run_validation_logs_both_validations(validated):
    folder, log, lines = validated
    table = read_log(log)
    assert set(table["validation"]) == {"walk_forward", "cpcv"}
    # default embargo 10 since Phase 16 (was 1), plus the sensitivity embargo 0
    assert set(table.loc[table["validation"] == "cpcv", "embargo"]) == {"0", "10"}
    assert (table["phase"] == "16").all() and table["experiment_id"].is_unique
    text = (folder / "validation.md").read_text()
    for part in ("## Moment: quali", "Embargo sensitivity", "## Gate decisions",
                 "Benjamini-Hochberg", "Trials per family", "Run times"):
        assert part in text
    assert (folder / "validation_cpcv_quali.png").exists()
    assert (folder / "validation_embargo.png").exists()
    assert any(line.startswith("CPCV quali embargo 0") for line in lines)
    evidence = load_evidence("B1 logistic", "quali", log, embargo=10)
    assert evidence.cpcv is not None and len(evidence.experiment_ids) == 2
    with pytest.raises(LockedSeasonError):
        load_evidence("B1 logistic", "quali", log, season=2026)
    with pytest.raises(Exception, match="No walk-forward run"):
        load_evidence("M9 nothing", "quali", log)


def test_cli_gate_report_only_and_merge(validated, capsys, tmp_path):
    folder, log, _ = validated
    code = cli.main(["gate", "B1 logistic", "B0 grid slot", "--moment", "quali",
                     "--log", str(log)])
    out = capsys.readouterr().out
    assert code == 0 and "gate against B0 grid slot" in out and "Trials in family" in out
    assert cli.main(["gate", "B1 logistic", "B0 grid slot", "--moment", "quali",
                     "--log", str(log), "--season", "2026"]) == 1
    assert "final test" in capsys.readouterr().out
    report = tmp_path / "again.md"
    assert cli.main(["validate", "--report-only", "--log", str(log),
                     "--report", str(report)]) == 0
    assert "## Gate decisions" in report.read_text()
    merged = tmp_path / "merged" / "experiments.csv"
    assert cli.main(["validate", "--merge", str(log), "--log", str(merged),
                     "--report", str(tmp_path / "merged" / "v.md")]) == 0
    assert len(read_log(merged)) == len(read_log(log))
    assert cli.main(["validate", "--report-only", "--log", str(tmp_path / "x.csv")]) == 3


def test_cli_validate_runs_on_a_dataset_file(data, tmp_path, capsys):
    pytest.importorskip("pyarrow")
    path = tmp_path / "d.parquet"
    data.to_parquet(path, index=False)
    log = tmp_path / "r" / "experiments.csv"
    code = cli.main(["validate", "--models", "B0 grid slot", "--moments", "pre",
                     "--sensitivity", "--no-pools", "--no-walk-forward", "--no-report",
                     "--dataset", str(path), "--log", str(log)])
    assert code == 0 and "CPCV pre embargo 1" in capsys.readouterr().out
    assert set(read_log(log)["validation"]) == {"cpcv"}
    assert cli.main(["validate", "--embargo", "-1", "--dataset", str(path)]) == 1


def test_pools_front_of_grid_and_report_sections(data, tmp_path):
    models = [GridSlotBaseline(), LogisticModel(n_sims=FAST),
              StrengthModel(params=WORLD_PARAMS, n_sims=FAST)]
    preds = walk_forward(models, data, "quali", seasons=[2024, 2025])
    log = tmp_path / "experiments.csv"
    lines = log_walk_forward_pools(preds, data, log, 2)
    assert lines == ["walk-forward pools: 12 configurations logged"]
    table = read_log(log)
    assert count_trials(table, "pool", "quali") == 12
    assert log_walk_forward_pools(preds[preds["Model"] != "M1 strength"], data, log, 2)[0] \
        .startswith("pools skipped")
    grid = data.loc[data["Moment"] == "quali", ["RaceId", "DriverId", "grid"]]
    stats_table = front_of_grid_stats(preds.merge(grid, on=["RaceId", "DriverId"]))
    assert set(stats_table["Model"]) == {"B0 grid slot", "B1 logistic", "M1 strength"}
    assert stats_table["Pole won"].between(0, 1).all()
    save_front_of_grid(preds, data, tmp_path)
    assert (tmp_path / "validation_front_of_grid.csv").exists()
    report = write_validation_report(log, tmp_path / "v.md", plots=False)
    text = report.read_text()
    assert "Why B1 / M1 lose to the grid" in text and "Probability of Backtest" in text
    assert "not changed in Phase 15" in text


def test_stage_temperatures_recover_a_planted_front():
    rng = np.random.default_rng(0)
    rows = []
    for race in range(200):
        s = rng.normal(0, 1, 12)
        keys = 1.0 * s + rng.gumbel(size=12)
        order = np.argsort(-keys)
        finish = np.empty(12)
        finish[order] = np.arange(1, 13)
        rows += [{"RaceId": race, "score": s[i], "Finish": finish[i]} for i in range(12)]
    table = pd.DataFrame(rows)
    betas = stage_temperatures(table["score"].to_numpy(), table["RaceId"].to_numpy(),
                               table["Finish"].to_numpy())
    assert betas.set_index("Places")["beta"].between(0.6, 1.5).all()   # true beta = 1


def test_helpers_and_recommendation():
    with pytest.raises(ValueError):
        model_factories_for(["B9 nothing"])
    factories = model_factories_for(["B0 grid slot"], include_m2=True, m2_seeds=1)
    assert set(factories) == {"B0 grid slot", "M2-full"}
    wide = pd.DataFrame({"Moment": ["pre", "quali"], "Model": ["B1", "M1"],
                         0: [0.140, 0.120], 1: [0.141, 0.121], 10: [0.1405, 0.1252]})
    wide["Spread"] = wide[[0, 1, 10]].max(axis=1) - wide[[0, 1, 10]].min(axis=1)
    text = " ".join(embargo_recommendation(wide, [0, 1, 10]))
    assert "larger default embargo" in text and "signature of leakage" in text
    calm = wide.assign(Spread=0.0001)
    assert "keep the default embargo" in " ".join(embargo_recommendation(calm, [0, 1, 10]))
    assert LoggedConfig("x", {"a": 1}).params() == {"a": 1}


def test_cpcv_race_losses_shape(data, plan):
    preds = run_cpcv([GridSlotBaseline()], data, "pre", plan).predictions
    table = cpcv_race_losses(preds)
    assert len(table) == 28 * 2 * 4 and set(table["Block"]) == set(range(8))
