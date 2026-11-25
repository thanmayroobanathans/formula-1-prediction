"""M1: the car/driver strength model (Kalman filter) as a RaceModel.

fit():
1. StrengthFeatures runs the Kalman filter over the training rows' OBSERVATIONS and
   gives every training row the filter's ONE-STEP-AHEAD estimate.
2. M1: linear Plackett-Luce on standardised [moment features + strength features].
   M1-pure: score = -predicted pace, only a temperature is fitted.
3. DNF: logistic regression, as in B1.
predict(): move the filter forward to the race date (no update: the race has not
happened), build the same features, simulate with the pace uncertainty as score_sd.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .._optional import require
from ..strength import StrengthParams
from ..strength_features import STRENGTH_FEATURES, StrengthFeatures
from .base import (
    RaceScores,
    standardise,
    N_SIMS, design_matrix, features_of, fit_binary, fit_scale, moment_of, predict_from_scores,
)
from .logistic import fit_linear_plackett_luce

__all__ = ["STRENGTH_FEATURES", "StrengthModel"]


class StrengthModel:
    """M1 (pure=False) or M1-pure (pure=True). Hyperparameters are fitted on training
    data only (complete seasons, cached), unless fixed `params` are given."""

    def __init__(self, pure: bool = False, l2: float = 1.0, dnf_c: float = 0.5,
                 n_sims: int = N_SIMS, seed: int = 0, params: StrengthParams | None = None,
                 maxiter: int = 60, use_score_sd: bool = True) -> None:
        self.name = "M1-pure strength" if pure else "M1 strength"
        self.pure = pure
        self.l2 = l2
        self.dnf_c = dnf_c
        self.n_sims = n_sims
        self.seed = seed
        self.fixed_params = params
        self.maxiter = maxiter
        self.use_score_sd = use_score_sd

    def params(self) -> dict[str, Any]:
        """Settings (and fitted hyperparameters, once fitted) for the experiment log."""
        out: dict[str, Any] = {"pure": self.pure, "l2": self.l2, "dnf_c": self.dnf_c,
                               "n_sims": self.n_sims, "seed": self.seed,
                               "maxiter": self.maxiter, "use_score_sd": self.use_score_sd}
        if hasattr(self, "params_"):
            out["strength"] = {k: round(v, 6) for k, v in asdict(self.params_).items()}
        return out

    def fit(self, train: pd.DataFrame) -> StrengthModel:
        """Filter the past, then learn how strengths (and features) turn into results."""
        preprocessing = require("sklearn.preprocessing", "M1 StrengthModel")
        linear = require("sklearn.linear_model", "M1 StrengthModel")
        self.moment_ = moment_of(train)
        self.strength_ = StrengthFeatures(self.fixed_params, self.maxiter)
        rows = self.strength_.fit(train)
        self.params_, self.filter_ = self.strength_.params_, self.strength_.filter_
        self.features_ = features_of(train) + ([] if self.pure else STRENGTH_FEATURES)
        raw = design_matrix(rows, self.features_)
        self.scaler_ = preprocessing.StandardScaler().fit(raw)
        z = standardise(self.scaler_, raw)
        races, finish = rows["RaceId"].to_numpy(), rows["Finish"].to_numpy(float)
        if self.pure:
            self.weights_ = np.zeros(0)
            self.scale_ = fit_scale(-rows["pace_pred"].to_numpy(float), races, finish)
        else:
            self.weights_ = fit_linear_plackett_luce(z, races, finish, self.l2)
            self.scale_ = 1.0
        dnf_model = linear.LogisticRegression(C=self.dnf_c, max_iter=2000)
        self.dnf_ = fit_binary(dnf_model, z, rows["DNF"].to_numpy())
        return self

    def race_scores(self, race: pd.DataFrame) -> RaceScores:
        """Raw scores (higher = faster), their uncertainty, DNF chances and temperature."""
        if moment_of(race) != self.moment_:
            raise ValueError(f"{self.name} was fitted on '{self.moment_}' rows.")
        rows = self.strength_features(race)
        z = standardise(self.scaler_, design_matrix(rows, self.features_))
        p_dnf = self.dnf_.predict_proba(z)[:, 1]
        if self.pure:
            scores = -rows["pace_pred"].to_numpy(float)
            spread = rows["pace_sd"].to_numpy(float)
        else:
            scores = z @ self.weights_
            spread = self._score_sd(rows)
        return RaceScores(scores, spread if self.use_score_sd else None, p_dnf, self.scale_)

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Probabilities for one race of the moment the model was fitted on."""
        s = self.race_scores(race)
        return predict_from_scores(race, s.scores, s.p_dnf, s.scale, self.n_sims, self.seed,
                                   s.spread)

    def strength_features(self, race: pd.DataFrame) -> pd.DataFrame:
        """The race rows plus the filter's forecast for each driver (same order)."""
        return self.strength_.for_race(race)

    def coefficients(self) -> pd.Series:
        """Ranking weights per standardised feature (M1) or the temperature (M1-pure)."""
        if self.pure:
            return pd.Series({"-pace_pred (temperature)": self.scale_})
        weights = pd.Series(self.weights_, index=self.features_)
        return weights.reindex(weights.abs().sort_values(ascending=False).index)

    def strengths(self) -> pd.DataFrame:
        """Latest car and driver estimates from the filter (lower Mean = faster)."""
        return self.strength_.strengths()

    def _score_sd(self, rows: pd.DataFrame) -> np.ndarray:
        """Score uncertainty caused by car/skill uncertainty (through their weights)."""
        i, j = self.features_.index("car_mean"), self.features_.index("skill_mean")
        a = self.weights_[i] / self.scaler_.scale_[i]
        b = self.weights_[j] / self.scaler_.scale_[j]
        var = (a**2 * rows["car_sd"] ** 2 + b**2 * rows["skill_sd"] ** 2
               + 2 * a * b * rows["car_skill_cov"])
        return np.sqrt(np.maximum(var.to_numpy(float), 0.0))
