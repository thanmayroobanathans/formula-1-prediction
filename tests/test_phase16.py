"""Phase 16: embargo default, stagewise Plackett-Luce (step 0) and M4 Bayesian model.
Offline, small synthetic data. The M4 tests need PyMC (the 'bayes' extra)."""
import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pitwall import cli
from pitwall.predict import validation
from pitwall.predict.models import LogisticModel, StagewiseModel, check_consistency
from pitwall.predict.models.base import (
    DEFAULT_STAGES, finish_tensor, fit_stage_multipliers, pl_loglik_and_grad,
    simulate_positions, simulate_positions_staged, stage_index, staged_pl_loglik,
)
from pitwall.predict.pipeline import model_factories
from synthetic_f1 import make_dataset

warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)


# ---------- step 0a: the embargo ----------
def test_default_embargo_is_ten_everywhere():
    assert validation.DEFAULT_EMBARGO == 10
    assert cli.DEFAULT_EMBARGO == validation.DEFAULT_EMBARGO
    args = cli.build_parser().parse_args(["validate"])
    assert args.embargo == 10 and 10 not in args.sensitivity
    assert 10 not in validation.SENSITIVITY_EMBARGOES


# ---------- step 0b: stagewise Plackett-Luce ----------
def test_stage_index():
    assert stage_index(12).tolist() == [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3]
    assert stage_index(5, (2,)).tolist() == [0, 0, 1, 1, 1]


def test_all_ones_is_ordinary_plackett_luce():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(6, 12))
    mask = np.ones_like(u, dtype=bool)
    mask[0, 10:] = False
    plain, _ = pl_loglik_and_grad(u, mask)
    assert staged_pl_loglik(u, mask, np.ones(4)) == pytest.approx(plain)


def test_staged_likelihood_by_hand():
    # 3 cars, finishing A, B, C with scores 1.0, 0.5, 0.0; stage 1 = 1st (multiplier 2)
    u = np.array([[1.0, 0.5, 0.0]])
    mask = np.ones_like(u, dtype=bool)
    m = np.array([2.0, 1.0])
    first = 2.0 - np.log(np.exp(2.0) + np.exp(1.0) + np.exp(0.0))
    second = 0.5 - np.log(np.exp(0.5) + np.exp(0.0))
    assert staged_pl_loglik(u, mask, m, stages=(1,)) == pytest.approx(first + second)


def test_staged_sampler_matches_plain_when_multipliers_are_one():
    scores = np.array([1.2, 0.6, 0.0, -0.4, -1.0])
    p_dnf = np.full(5, 0.05)
    plain = simulate_positions(scores, p_dnf, 60_000, np.random.default_rng(1))
    staged = simulate_positions_staged(scores, p_dnf, 60_000, np.random.default_rng(2),
                                       np.ones(4))
    assert np.abs(plain - staged).max() < 0.01
    assert staged.sum(axis=0) == pytest.approx(np.ones(5), abs=1e-6)
    assert staged.sum(axis=1) == pytest.approx(np.ones(5), abs=1e-6)


def test_sharper_front_gives_the_favourite_more():
    scores = np.linspace(1.0, -1.0, 10)
    plain = simulate_positions_staged(scores, np.zeros(10), 40_000,
                                      np.random.default_rng(3), np.ones(4))
    sharp = simulate_positions_staged(scores, np.zeros(10), 40_000,
                                      np.random.default_rng(3), np.array([3.0, 1, 1, 1]))
    assert sharp[0, 0] > plain[0, 0] + 0.1


def _orders_from(true_m, n_races=400, n=12, seed=0):
    rng = np.random.default_rng(seed)
    scores, races, finish = [], [], []
    for r in range(n_races):
        s = rng.normal(0, 1, n)
        pos = simulate_positions_staged(s, np.zeros(n), 1, rng, true_m)
        place = pos.argmax(axis=1) + 1
        scores.append(s)
        races.append(np.full(n, r))
        finish.append(place.astype(float))
    return np.concatenate(scores), np.concatenate(races), np.concatenate(finish)


def test_fit_recovers_stage_multipliers():
    true_m = np.array([2.5, 1.5, 1.0, 0.3])
    fitted = fit_stage_multipliers(*_orders_from(true_m))
    assert fitted == pytest.approx(true_m, rel=0.4)
    assert fitted[0] > fitted[2] > fitted[3]


def test_monotone_option_never_increases():
    true_m = np.array([0.6, 1.4, 1.0, 0.5])   # deliberately not monotone
    fitted = fit_stage_multipliers(*_orders_from(true_m, n_races=150), monotone=True)
    assert all(a >= b - 1e-9 for a, b in zip(fitted, fitted[1:]))


def test_stagewise_model_fits_out_of_fold_and_predicts():
    data = make_dataset(seasons=(2023, 2024, 2025), races_per_season=8, n_drivers=10)
    rows = data[data["Moment"] == "quali"]
    train, race = rows[rows["Year"] < 2025], rows[rows["RaceId"] == rows["RaceId"].max()]
    calls = []

    def base():
        calls.append(1)
        return LogisticModel(n_sims=2000)

    model = StagewiseModel(base, "B1 stagewise", n_sims=4000).fit(train)
    assert len(calls) == 1 + model.folds          # full model + one per fold
    assert len(model.multipliers_) == len(DEFAULT_STAGES) + 1
    pred = model.predict(race)
    check_consistency(pred)
    assert model.params()["multipliers"] and "base" in model.params()


def test_stagewise_models_are_registered():
    known = model_factories(n_seeds=1)
    for name in ("B1 stagewise", "M1 stagewise", "M4 Bayesian", "M4 stagewise"):
        assert name in known and name in cli.MODEL_CHOICES
    assert known["B1 stagewise"]().name == "B1 stagewise"


def test_finish_tensor_used_by_stage_fit_skips_dnfs():
    tensor, mask = finish_tensor(np.arange(4.0), np.zeros(4), np.array([1, np.nan, 2, 3.0]))
    assert mask.sum() == 3


# ---------- M4: the Bayesian model ----------
pymc = pytest.importorskip("pymc")
from pitwall.predict.models.bayes import (  # noqa: E402
    BayesFitError, BayesianModel, build_layout, check_advi, check_nuts,
)
from pitwall.predict.strength import observation_table  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return make_dataset(seasons=(2022, 2023, 2024, 2025), races_per_season=8, n_drivers=12)


@pytest.fixture(scope="module")
def fitted(data):
    rows = data[(data["Moment"] == "quali") & (data["Year"] < 2025)]
    return BayesianModel(n_iter=6000, n_draws=150, n_sims=4000).fit(rows)


def test_layout_blocks_never_cross_seasons(data):
    obs = observation_table(data[data["Moment"] == "pre"])
    lay = build_layout(obs, block=3)
    years_per_block = pd.Series(lay.years).groupby(lay.block_of_race).nunique()
    assert (years_per_block == 1).all()
    assert lay.block_start.sum() == len(lay.seasons)
    assert lay.block_era[0] == (lay.years[0] in (2014, 2017, 2022, 2026))


def test_posterior_recovers_the_true_order(data, fitted):
    race = data[(data["Moment"] == "quali") & (data["Year"] == 2025)]
    race = race[race["RaceId"] == race["RaceId"].min()]
    pace = fitted.forecast_pace(race).mean(axis=0)
    true_skill = -race["DriverId"].str.extract(r"(\d+)")[0].astype(int).to_numpy()
    rho = stats.spearmanr(pace, -true_skill).correlation   # lower pace = faster = better
    assert rho > 0.7
    check_consistency(fitted.predict(race))
    s = fitted.race_scores(race)
    assert np.all(s.spread > 0) and s.scale == 1.0


def test_rookies_and_rule_resets_are_more_uncertain(data, fitted):
    race = data[(data["Moment"] == "quali") & (data["Year"] == 2025)].copy()
    race = race[race["RaceId"] == race["RaceId"].min()].reset_index(drop=True)
    rookie = race.copy()
    rookie.loc[0, "DriverId"] = "brand-new-driver"
    sd_known = fitted.forecast_pace(race).std(axis=0)[0]
    sd_rookie = fitted.forecast_pace(rookie).std(axis=0)[0]
    assert sd_rookie > sd_known
    later = race.assign(Year=2027, Date=race["Date"] + pd.Timedelta(days=730))
    reset = race.assign(Year=2026, Date=race["Date"] + pd.Timedelta(days=730))
    assert fitted.forecast_pace(reset).std(axis=0).mean() >= \
        fitted.forecast_pace(later).std(axis=0).mean()


def test_quali_gaps_update_the_forecast(data, fitted):
    race = data[(data["Moment"] == "quali") & (data["Year"] == 2025)]
    race = race[race["RaceId"] == race["RaceId"].min()].reset_index(drop=True)
    flipped = race.copy()
    flipped["quali_gap_pct"] = race["quali_gap_pct"].to_numpy()[::-1]
    a, b = fitted.race_scores(race).scores, fitted.race_scores(flipped).scores
    assert not np.allclose(a, b)
    assert fitted.race_scores(race).spread.mean() < np.sqrt(
        fitted.forecast_pace(race).var(axis=0)).mean() * float(fitted.hyper_["beta"].mean())


def test_no_future_data_is_used(data):
    rows = data[(data["Moment"] == "pre") & (data["Year"] < 2025)]
    target = data[(data["Moment"] == "pre") & (data["Year"] == 2025)]
    target = target[target["RaceId"] == target["RaceId"].min()]
    model = BayesianModel(n_iter=4000, n_draws=80, n_sims=2000)
    first = model.fit(rows).predict(target)
    scrambled = target.assign(Finish=np.nan, Win=0, QualiGapObs=9.0, PracticeGapObs=9.0)
    second = model.predict(scrambled)
    assert np.allclose(first["P_win"], second["P_win"])


def test_strengths_table_and_plot(fitted):
    table = fitted.strengths()
    assert {"car", "driver"} == set(table["Kind"])
    assert (table["Low"] <= table["Mean"]).all() and (table["Mean"] <= table["High"]).all()
    import matplotlib
    matplotlib.use("Agg")
    from pitwall.plotting import strength_intervals
    assert strength_intervals(table, "car").get_title() == "Car pace"


def _idata(az, posterior):
    stats_ = {"diverging": np.zeros((2, 200), bool)}
    if int(az.__version__.split(".")[0]) >= 1:      # ArviZ 1.x (Python >= 3.12)
        return az.from_dict({"posterior": posterior, "sample_stats": stats_})
    return az.from_dict(posterior=posterior, sample_stats=stats_)   # ArviZ 0.x


def test_advi_and_nuts_checks():
    with pytest.raises(BayesFitError, match="NaN"):
        check_advi(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(BayesFitError, match="converge"):
        check_advi(np.concatenate([np.full(80, 1000.0), np.full(10, 800.0), np.full(10, 500.0)]))
    check_advi(np.concatenate([np.linspace(0, -30, 60), -30 + np.random.default_rng(1).normal(
        0, 3, 40)]))  # small, noisy loss that has settled
    check_advi(np.concatenate([np.linspace(2000, 600, 50), np.full(50, 500.0)]))
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(0)
    good = {k: rng.normal(1, 0.1, (2, 200)) for k in ("s_car", "s_winter", "s_skill",
                                                       "s_quali", "beta")}
    draws = _idata(az, good)
    assert check_nuts(draws)["divergences"] == 0
    bad = dict(good, beta=np.stack([rng.normal(0, 0.1, 200), rng.normal(5, 0.1, 200)]))
    with pytest.raises(BayesFitError, match="R-hat"):
        check_nuts(_idata(az, bad))


def test_bad_method_is_refused():
    with pytest.raises(ValueError):
        BayesianModel(method="mcmc-please")


def test_cli_strengths(tmp_path, data, capsys):
    path = tmp_path / "d.parquet"
    data.to_parquet(path)
    assert cli.main(["strengths", "2024", "--after-round", "4", "--dataset", str(path),
                     "--iterations", "3000", "--plot", str(tmp_path / "cars.png")]) == 0
    out = capsys.readouterr().out
    assert "Car pace" in out and "Driver skill" in out and (tmp_path / "cars.png").exists()


def test_advi_continues_once_before_giving_up(data, monkeypatch):
    import pitwall.predict.models.bayes as bayes
    calls = []

    def strict_then_fine(history, **_):
        calls.append(len(history))
        if len(calls) == 1:
            raise BayesFitError("ADVI did not converge (test)")

    monkeypatch.setattr(bayes, "check_advi", strict_then_fine)
    rows = data[(data["Moment"] == "pre") & (data["Year"] == 2024)]
    BayesianModel(n_iter=200, n_draws=30, n_sims=500).fit(rows)
    assert calls == [200, 400]                  # refined once, from where it stopped

    def always_nan(history, **_):
        raise BayesFitError("ADVI produced NaN or inf: test")

    monkeypatch.setattr(bayes, "check_advi", always_nan)
    with pytest.raises(BayesFitError, match="NaN"):
        BayesianModel(n_iter=50, n_draws=30, n_sims=500).fit(rows)


def test_stagewise_is_its_own_family_and_phase_16_is_logged():
    from pitwall.predict.experiments import PHASE, is_validated_phase, model_family
    assert PHASE == "16"
    assert model_family("B1 stagewise") == model_family("M1 stagewise") == "stagewise PL"
    assert model_family("B1 logistic") == "B1" and model_family("M4 Bayesian") == "M4"
    assert is_validated_phase(pd.Series(["<=14", "15", "16", ""])).tolist() == \
        [False, True, True, False]
