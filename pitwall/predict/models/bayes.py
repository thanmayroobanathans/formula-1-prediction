"""Phase 16: M4, a hierarchical Bayesian car/driver model (PyMC).

The Kalman filter (M1) keeps ONE best guess per car and driver, plus an error bar that
assumes everything is Gaussian. M4 writes down the whole story as one probability
model and lets PyMC find every quantity that fits the past at once, with honest
uncertainty:

    pace[d, r]  = car[team(d, r), block(r)] + skill[d, season(r)]        (% gap, lower = faster)

    car         Gaussian random walk over blocks of BLOCK races; a bigger jump at every
                season start (s_winter) and a bigger one still in a rule-reset season
                (s_era: 2014, 2017, 2022, 2026). "Rules changed, so last year means less."
    skill       random walk over seasons (s_skill). A driver's FIRST season has a
                hierarchical prior: veterans (already racing when the data starts) are
                N(0, s_vet); real rookies are N(mu_rookie, s_rookie), and mu_rookie and
                s_rookie are learned from all past rookies. So a debutant is UNCERTAIN,
                not "average" (the Senna-at-Monaco-1984 case).
    data        qualifying gaps  ~ Normal(pace, s_quali)
                practice gaps    ~ Normal(pace, s_practice)     (2023+, OpenF1)
                finishing order  ~ Plackett-Luce(-beta * pace)  (classified finishers)

Team-mates share the car term, so their difference is pure skill: that is how car and
driver are told apart (as in M1). Inference: ADVI (fast, variational) by default; NUTS
(MCMC) as an option. Diagnostics are checked and a bad fit raises BayesFitError.

Predicting a race: posterior draws of the last car/skill values are moved forward to
the race date (more drift if time passed, the season / era jumps if a new season
started, the rookie prior for a new driver), then, at the practice / quali moment,
updated with that race's own practice / qualifying gaps (a Gaussian update per driver).
Scores = -beta * pace; their posterior spread becomes score_sd, so the 20,000
simulations are wider exactly where the model is unsure. DNFs: logistic model on the
moment's features, as in B1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._optional import require
from ...errors import PitwallError
from ..features import ERA_STARTS
from ..strength import NEW_CAR_VAR, observation_table
from .base import (
    N_SIMS, RaceScores, design_matrix, features_of, fit_binary, moment_of,
    predict_from_scores, standardise,
)

log = logging.getLogger(__name__)
BLOCK = 3                 # car strength is constant within blocks of 3 races
DAYS_PER_BLOCK = 42.0     # ~3 races at 14 days
PAD = -1e9                # padding score in the Plackett-Luce likelihood


class BayesFitError(PitwallError):
    """The Bayesian fit did not converge (NaN, diverging, or bad R-hat / divergences)."""


@dataclass(frozen=True)
class Layout:
    """Index arrays that turn observation rows into model coordinates."""

    races: np.ndarray          # race ids, date order
    dates: np.ndarray
    years: np.ndarray          # season (year) per race
    block_of_race: np.ndarray  # block index per race
    block_start: np.ndarray    # 1 if the block opens a season
    block_era: np.ndarray      # 1 if the block opens a rule-reset season
    seasons: np.ndarray        # distinct years, sorted
    teams: list[str]
    drivers: list[str]
    first_season: np.ndarray   # season index of each driver's first race
    rookie: np.ndarray         # 1 if the driver's first season is after the data starts


def build_layout(obs: pd.DataFrame, block: int = BLOCK) -> Layout:
    """Blocks of `block` races that never cross a season boundary."""
    races = obs.drop_duplicates("RaceId").sort_values(["Date", "RaceId"])
    years = races["Year"].to_numpy(int)
    block_of = np.zeros(len(races), dtype=int)
    starts, eras = [1], [1 if years[0] in ERA_STARTS else 0]
    count = 0
    for i in range(1, len(races)):
        new_season = years[i] != years[i - 1]
        count += 1
        if new_season or count >= block:
            block_of[i] = block_of[i - 1] + 1
            starts.append(1 if new_season else 0)
            eras.append(1 if new_season and years[i] in ERA_STARTS else 0)
            count = 0
        else:
            block_of[i] = block_of[i - 1]
    seasons = np.unique(years)
    drivers = sorted(obs["DriverId"].astype(str).unique())
    first_year = obs.groupby(obs["DriverId"].astype(str))["Year"].min()
    first_season = np.searchsorted(seasons, first_year.reindex(drivers).to_numpy())
    rookie = (first_year.reindex(drivers).to_numpy() > seasons[0]).astype(float)
    return Layout(races["RaceId"].to_numpy(), races["Date"].to_numpy(), years, block_of,
                  np.array(starts, float), np.array(eras, float), seasons,
                  sorted(obs["TeamId"].astype(str).unique()), drivers, first_season, rookie)


def finishing_orders(rows: pd.DataFrame, races: np.ndarray, row_index: pd.Series
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(order, mask): order[r, k] = observation row of the k-th classified finisher."""
    groups = []
    for race_id in races:
        race = rows[(rows["RaceId"] == race_id) & rows["Finish"].notna()]
        race = race.sort_values("Finish")
        idx = [row_index.get((race_id, str(d))) for d in race["DriverId"]]
        idx = [i for i in idx if i is not None]
        groups.append(idx)
    width = max((len(g) for g in groups), default=0)
    order = np.zeros((len(groups), max(width, 1)), dtype=int)
    mask = np.zeros_like(order, dtype=bool)
    for r, idx in enumerate(groups):
        order[r, :len(idx)] = idx
        mask[r, :len(idx)] = True
    return order, mask


class BayesianModel:
    """M4: hierarchical Bayesian car/driver strengths -> Plackett-Luce -> Monte Carlo."""

    def __init__(self, method: str = "advi", n_iter: int = 15_000, n_draws: int = 200,
                 block: int = BLOCK, dnf_c: float = 0.5, n_sims: int = N_SIMS, seed: int = 0,
                 nuts_tune: int = 500, chains: int = 2, learning_rate: float = 0.01) -> None:
        if method not in ("advi", "nuts"):
            raise ValueError("method must be 'advi' or 'nuts'.")
        self.name = "M4 Bayesian"
        self.method = method
        self.n_iter = n_iter
        self.n_draws = n_draws
        self.block = block
        self.dnf_c = dnf_c
        self.n_sims = n_sims
        self.seed = seed
        self.nuts_tune = nuts_tune
        self.chains = chains
        self.learning_rate = learning_rate

    def params(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method, "n_iter": self.n_iter,
                               "learning_rate": self.learning_rate,
                               "n_draws": self.n_draws, "block": self.block,
                               "dnf_c": self.dnf_c, "n_sims": self.n_sims, "seed": self.seed}
        if hasattr(self, "hyper_"):
            out["posterior_means"] = {k: round(float(np.mean(v)), 4)
                                      for k, v in self.hyper_.items()}
        return out

    # ----- fitting -----
    def fit(self, train: pd.DataFrame) -> BayesianModel:
        preprocessing = require("sklearn.preprocessing", "M4 BayesianModel")
        linear = require("sklearn.linear_model", "M4 BayesianModel")
        self.moment_ = moment_of(train)
        obs = observation_table(train)
        self.layout_ = build_layout(obs, self.block)
        self._fit_posterior(train, obs)
        self.features_ = features_of(train)
        raw = design_matrix(train, self.features_)
        self.scaler_ = preprocessing.StandardScaler().fit(raw)
        z = standardise(self.scaler_, raw)
        dnf_model = linear.LogisticRegression(C=self.dnf_c, max_iter=2000)
        self.dnf_ = fit_binary(dnf_model, z, train["DNF"].to_numpy())
        return self

    def _fit_posterior(self, train: pd.DataFrame, obs: pd.DataFrame) -> None:
        pm = require("pymc", "M4 BayesianModel", extra="bayes")
        pt = require("pytensor.tensor", "M4 BayesianModel", extra="bayes")
        lay = self.layout_
        race_pos = {int(r): i for i, r in enumerate(lay.races)}
        team_pos = {t: i for i, t in enumerate(lay.teams)}
        driver_pos = {d: i for i, d in enumerate(lay.drivers)}
        ri = obs["RaceId"].map(race_pos).to_numpy(int)
        ti = obs["TeamId"].astype(str).map(team_pos).to_numpy(int)
        di = obs["DriverId"].astype(str).map(driver_pos).to_numpy(int)
        bi = lay.block_of_race[ri]
        yi = np.searchsorted(lay.seasons, lay.years[ri])
        row_index = pd.Series(np.arange(len(obs)),
                              index=pd.MultiIndex.from_arrays(
                                  [obs["RaceId"].to_numpy(), obs["DriverId"].astype(str)]))
        order, mask = finishing_orders(train, lay.races, row_index)
        quali = obs["QualiObs"].to_numpy(float)
        practice = obs["PracticeObs"].to_numpy(float)
        q_ok, p_ok = np.isfinite(quali), np.isfinite(practice)
        n_blocks, n_seasons = int(lay.block_of_race.max()) + 1, len(lay.seasons)
        n_teams, n_drivers = len(lay.teams), len(lay.drivers)
        later = (np.arange(n_seasons)[None, :] > lay.first_season[:, None]).astype(float)
        with pm.Model() as model:
            s_car = pm.HalfNormal("s_car", 0.2)
            s_winter = pm.HalfNormal("s_winter", 0.5)
            s_era = pm.HalfNormal("s_era", 1.0)
            s_skill = pm.HalfNormal("s_skill", 0.2)
            s_vet = pm.HalfNormal("s_vet", 0.5)
            mu_rookie = pm.Normal("mu_rookie", 0.2, 0.3)
            s_rookie = pm.HalfNormal("s_rookie", 0.5)
            s_quali = pm.HalfNormal("s_quali", 0.5)
            s_practice = pm.HalfNormal("s_practice", 1.0)
            beta = pm.HalfNormal("beta", 3.0)
            car_z = pm.Normal("car_z", 0, 1, shape=(n_teams, n_blocks))
            step_sd = pt.sqrt(s_car**2 + lay.block_start * s_winter**2
                              + lay.block_era * s_era**2)
            car = pm.Deterministic("car", pt.cumsum(car_z * step_sd[None, :], axis=1))
            skill_z = pm.Normal("skill_z", 0, 1, shape=(n_drivers, n_seasons))
            first_z = skill_z[np.arange(n_drivers), lay.first_season]
            base = lay.rookie * (mu_rookie + s_rookie * first_z) + (1 - lay.rookie) * (
                s_vet * first_z)
            skill = pm.Deterministic(
                "skill", base[:, None] + pt.cumsum(skill_z * s_skill * later, axis=1))
            pace = car[ti, bi] + skill[di, yi]
            if q_ok.any():
                pm.Normal("quali", pace[q_ok], s_quali, observed=quali[q_ok])
            if p_ok.any():
                pm.Normal("practice", pace[p_ok], s_practice, observed=practice[p_ok])
            if mask.any():
                u = pt.switch(mask, -beta * pace[order], PAD)
                top = u.max(axis=1, keepdims=True)
                e = pt.exp(u - top)
                tail = pt.flip(pt.cumsum(pt.flip(e, axis=1), axis=1), axis=1)
                # padded places have tail = 0: add 1 there so log() stays finite, then
                # drop those terms with switch (0 * inf would be NaN)
                safe = tail + (~mask).astype(float)
                pm.Potential("order", pt.sum(pt.switch(mask, u - top - pt.log(safe), 0.0)))
            draws = self._infer(pm, model)
        post = draws.posterior
        stack = {k: post[k].stack(s=("chain", "draw")).transpose("s", ...).to_numpy()
                 for k in ("s_car", "s_winter", "s_era", "s_skill", "mu_rookie", "s_rookie",
                           "s_quali", "s_practice", "beta", "car", "skill")}
        self.hyper_ = {k: stack[k] for k in ("s_car", "s_winter", "s_era", "s_skill",
                                             "mu_rookie", "s_rookie", "s_quali",
                                             "s_practice", "beta")}
        self.car_last_ = stack["car"][:, :, -1]          # (draws, teams)
        self.car_all_ = stack["car"]
        self.skill_last_ = stack["skill"][:, :, -1]      # (draws, drivers)
        self.skill_all_ = stack["skill"]
        self.last_date_ = pd.Timestamp(lay.dates[-1])
        self.last_year_ = int(lay.years[-1])
        last_team = obs.sort_values("Date").groupby(obs["DriverId"].astype(str))["TeamId"].last()
        self.last_team_ = last_team.astype(str).to_dict()

    def _infer(self, pm: Any, model: Any) -> Any:
        if self.method == "advi":
            advi = pm.ADVI(random_seed=self.seed)
            optimiser = pm.adam(learning_rate=self.learning_rate)
            approx = advi.fit(self.n_iter, progressbar=False, obj_optimizer=optimiser)
            history = np.asarray(advi.hist, float)
            try:
                check_advi(history)
            except BayesFitError as err:
                if "NaN" in str(err):
                    raise
                # Still improving: continue the same run once (same state), then judge.
                log.info("ADVI still improving after %d iterations; continuing.", self.n_iter)
                advi.refine(self.n_iter, progressbar=False)   # updates advi.approx in place
                approx = advi.approx
                history = np.asarray(advi.hist, float)          # both runs' losses
                check_advi(history)
            self.diagnostics_ = {"final_loss": float(np.mean(history[-max(1, len(history)
                                                                         // 20):]))}
            return approx.sample(self.n_draws, random_seed=self.seed)
        draws = pm.sample(self.n_draws, tune=self.nuts_tune, chains=self.chains, cores=1,
                          progressbar=False, random_seed=self.seed)
        self.diagnostics_ = check_nuts(draws)
        return draws

    # ----- predicting -----
    def race_scores(self, race: pd.DataFrame) -> RaceScores:
        if moment_of(race) != self.moment_:
            raise ValueError(f"{self.name} was fitted on '{self.moment_}' rows.")
        pace = self.forecast_pace(race)                       # (draws, drivers)
        mean, var = pace.mean(axis=0), pace.var(axis=0)
        mean, var = self._condition(race, mean, var)
        beta = float(np.mean(self.hyper_["beta"]))
        z = standardise(self.scaler_, design_matrix(race, self.features_))
        p_dnf = self.dnf_.predict_proba(z)[:, 1]
        return RaceScores(-beta * mean, beta * np.sqrt(var), p_dnf, 1.0)

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        s = self.race_scores(race)
        return predict_from_scores(race, s.scores, s.p_dnf, s.scale, self.n_sims, self.seed,
                                   s.spread)

    def forecast_pace(self, race: pd.DataFrame) -> np.ndarray:
        """Posterior draws of each driver's pace at this race, BEFORE its own data."""
        rng = np.random.default_rng([self.seed, int(race["RaceId"].iloc[0])])
        h = self.hyper_
        n_draws = len(h["beta"])
        date = pd.Timestamp(race["Date"].iloc[0])
        year = int(race["Year"].iloc[0])
        steps = max(0.0, (date - self.last_date_).days / DAYS_PER_BLOCK)
        new_season = year > self.last_year_
        car_var = steps * h["s_car"] ** 2 + new_season * h["s_winter"] ** 2 + (
            new_season and year in ERA_STARTS) * h["s_era"] ** 2
        team_pos = {t: i for i, t in enumerate(self.layout_.teams)}
        driver_pos = {d: i for i, d in enumerate(self.layout_.drivers)}
        out = np.empty((n_draws, len(race)))
        for j, (team, driver) in enumerate(zip(race["TeamId"].astype(str),
                                               race["DriverId"].astype(str))):
            noise = rng.standard_normal((2, n_draws))
            if team in team_pos:
                car = self.car_last_[:, team_pos[team]] + np.sqrt(car_var) * noise[0]
            else:
                car = np.sqrt(NEW_CAR_VAR) * noise[0]
            if driver in driver_pos:
                skill = self.skill_last_[:, driver_pos[driver]] + (
                    new_season * h["s_skill"] * noise[1])
            else:
                skill = h["mu_rookie"] + h["s_rookie"] * noise[1]
            out[:, j] = car + skill
        return out

    def _condition(self, race: pd.DataFrame, mean: np.ndarray,
                   var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian update with this race's own gaps when the moment has them."""
        updates = []
        if self.moment_ in ("practice", "quali") and "practice_pace_gap_pct" in race:
            has = race.get("has_practice_pace", pd.Series(0, index=race.index))
            if (pd.to_numeric(has, errors="coerce").fillna(0) > 0).any():
                updates.append((race["practice_pace_gap_pct"], self.hyper_["s_practice"]))
        if self.moment_ == "quali" and "quali_gap_pct" in race:
            updates.append((race["quali_gap_pct"], self.hyper_["s_quali"]))
        for values, sigma in updates:
            y = pd.to_numeric(values, errors="coerce").to_numpy(float)
            y = y - np.nanmedian(y)                     # centred, like the training gaps
            noise = float(np.mean(sigma)) ** 2
            ok = np.isfinite(y)
            post_var = np.where(ok, 1.0 / (1.0 / np.maximum(var, 1e-9) + 1.0 / noise), var)
            mean = np.where(ok, post_var * (mean / np.maximum(var, 1e-9)
                                            + np.nan_to_num(y) / noise), mean)
            var = post_var
        return mean, var

    def strengths(self, level: float = 0.8) -> pd.DataFrame:
        """Latest posterior car and driver strengths with a `level` interval
        (lower = faster, in % gap)."""
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        rows = []
        for kind, names, values in (("car", self.layout_.teams, self.car_last_),
                                    ("driver", self.layout_.drivers, self.skill_last_)):
            for i, name in enumerate(names):
                v = values[:, i]
                rows.append({"Kind": kind, "Id": name, "Mean": float(v.mean()),
                             "Low": float(np.quantile(v, lo)), "High": float(np.quantile(v, hi)),
                             "Rookie": bool(kind == "driver" and self.layout_.rookie[i])})
        return pd.DataFrame(rows).sort_values(["Kind", "Mean"]).reset_index(drop=True)


def check_advi(history: np.ndarray, window: float = 0.1, tolerance: float = 0.05) -> None:
    """Raise BayesFitError if ADVI produced NaN or was still improving at the end.

    'Still improving' = the last 10% of iterations improved the loss by more than 5% of
    the total improvement since the start AND by more than 2 standard deviations of the
    loss noise in that window (ADVI's loss is noisy, so a tiny drift is not a failure).
    """
    if not np.all(np.isfinite(history)):
        raise BayesFitError("ADVI produced NaN or inf: the fit diverged.")
    n = max(2, int(len(history) * window))
    if len(history) < 3 * n:
        return
    start, before, last = (history[:n].mean(), history[-2 * n:-n].mean(),
                           history[-n:].mean())
    recent, total = before - last, abs(start - last)
    noise = history[-n:].std()
    if recent > tolerance * total and recent > 2 * noise:
        raise BayesFitError(f"ADVI did not converge (loss {before:.1f} -> {last:.1f} in the "
                            "last 10% of iterations); increase n_iter.")


def _values(summary: Any) -> np.ndarray:
    """All numbers in an ArviZ rhat/ess result (xarray Dataset or ArviZ 1.x DataTree)."""
    node = getattr(summary, "dataset", summary)     # DataTree -> its Dataset
    return np.concatenate([np.ravel(np.asarray(node[name], float)) for name in node.data_vars])


def check_nuts(draws: Any, max_rhat: float = 1.05) -> dict[str, float]:
    """Raise BayesFitError on divergences or R-hat above max_rhat (hyperparameters)."""
    az = require("arviz", "M4 BayesianModel", extra="bayes")
    divergences = int(np.asarray(draws["sample_stats"]["diverging"]).sum())
    names = ["s_car", "s_winter", "s_skill", "s_quali", "beta"]
    rhat = float(np.nanmax(_values(az.rhat(draws, var_names=names))))
    ess = float(np.nanmin(_values(az.ess(draws, var_names=names))))
    if divergences > 0 or not np.isfinite(rhat) or rhat > max_rhat:
        raise BayesFitError(f"NUTS diagnostics failed: {divergences} divergences, "
                            f"max R-hat {rhat:.3f} (want 0 and <= {max_rhat}).")
    return {"divergences": divergences, "max_rhat": rhat, "min_ess": ess}
