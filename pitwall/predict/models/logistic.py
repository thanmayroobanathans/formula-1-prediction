"""B1: a linear Plackett-Luce ranking model plus a logistic DNF model.

Why this design:
- We want one strength per driver from a weighted sum of features, fitted to the
  whole finishing order, not just "won / didn't win". That is the rank-ordered
  ('conditional') logit: Plackett-Luce with score = features . weights. With two
  drivers it is exactly logistic regression on their feature difference, so it is the
  natural multi-driver version of logistic regression.
- scikit-learn has no rank-ordered logit, so the weights are fitted with SciPy's
  L-BFGS on the Plackett-Luce likelihood (with a small L2 penalty). The likelihood
  learns the overall size of the weights too, so no separate temperature is needed.
- scikit-learn standardises the features and fits the separate DNF logistic model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from .._optional import require
from .base import (
    RaceScores,
    standardise,
    N_SIMS, design_matrix, features_of, finish_tensor, fit_binary, moment_of,
    pl_loglik_and_grad, predict_from_scores,
)


def fit_linear_plackett_luce(features: np.ndarray, races: np.ndarray, finish: np.ndarray,
                             l2: float = 1.0) -> np.ndarray:
    """Weights w maximising the Plackett-Luce likelihood of score = features @ w.

    Only classified finishers count (DNFs are the DNF model's job).
    """
    tensor, mask = finish_tensor(features, races, finish)
    size = features.shape[1]
    if not mask.any():
        return np.zeros(size)

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        loglik, grad_u = pl_loglik_and_grad(tensor @ w, mask)
        grad_w = np.einsum("rn,rnd->d", grad_u, tensor)
        return -loglik + 0.5 * l2 * float(w @ w), -grad_w + l2 * w

    result = optimize.minimize(loss, np.zeros(size), jac=True, method="L-BFGS-B")
    return np.asarray(result.x, float)


class LogisticModel:
    """B1: standardised features -> linear Plackett-Luce strengths -> Monte Carlo.

    P_dnf comes from a separate logistic regression on the same features.
    """

    name = "B1 logistic"

    def __init__(self, l2: float = 1.0, dnf_c: float = 0.5, n_sims: int = N_SIMS,
                 seed: int = 0) -> None:
        self.l2 = l2          # L2 penalty on the ranking weights (bigger = more cautious)
        self.dnf_c = dnf_c    # inverse L2 strength of the DNF logistic regression
        self.n_sims = n_sims
        self.seed = seed

    def params(self) -> dict[str, Any]:
        """Settings logged with each experiment."""
        return {"l2": self.l2, "dnf_c": self.dnf_c, "n_sims": self.n_sims, "seed": self.seed}

    def fit(self, train: pd.DataFrame) -> LogisticModel:
        """Standardise, fit the ranking weights, then the DNF model."""
        preprocessing = require("sklearn.preprocessing", "B1 LogisticModel")
        linear = require("sklearn.linear_model", "B1 LogisticModel")
        self.moment_ = moment_of(train)
        self.features_ = features_of(train)
        raw = design_matrix(train, self.features_)
        self.scaler_ = preprocessing.StandardScaler().fit(raw)
        z = standardise(self.scaler_, raw)
        self.weights_ = fit_linear_plackett_luce(
            z, train["RaceId"].to_numpy(), train["Finish"].to_numpy(float), self.l2)
        dnf_model = linear.LogisticRegression(C=self.dnf_c, max_iter=2000)
        self.dnf_ = fit_binary(dnf_model, z, train["DNF"].to_numpy())
        return self

    def race_scores(self, race: pd.DataFrame) -> RaceScores:
        """Raw scores (higher = faster), their uncertainty, DNF chances and temperature."""
        if moment_of(race) != self.moment_:
            raise ValueError(f"{self.name} was fitted on '{self.moment_}' rows.")
        z = standardise(self.scaler_, design_matrix(race, self.features_))
        return RaceScores(z @ self.weights_, None, self.dnf_.predict_proba(z)[:, 1], 1.0)

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Probabilities for one race of the moment the model was fitted on."""
        s = self.race_scores(race)
        return predict_from_scores(race, s.scores, s.p_dnf, s.scale, self.n_sims, self.seed)

    def coefficients(self) -> pd.Series:
        """Ranking weight per standardised feature (+ = helps finish ahead), biggest first."""
        weights = pd.Series(self.weights_, index=self.features_)
        return weights.reindex(weights.abs().sort_values(ascending=False).index)
