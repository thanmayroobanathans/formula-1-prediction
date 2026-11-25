"""Phase 16 step 0: stagewise Plackett-Luce, "sharper at the front".

Phase 15 found why B1 and M1 lose to the plain grid on the winner: Plackett-Luce turns
scores into probabilities with ONE temperature for every place, fitted on whole
orders. The front of an F1 race is predictable (pole won 67% of 2025 races) and the
back is chaos, so that single temperature is a compromise that is too flat at the front
(the pole-sitter got ~21% from B1/M1).

StagewiseModel wraps any score model (anything with race_scores(), e.g. B1, M1, M4)
and gives each stage of the order its own temperature multiplier:
    1st | 2nd-3rd | 4th-10th | 11th onwards  (DEFAULT_STAGES)
The multipliers are fitted by maximum likelihood on OUT-OF-FOLD scores (races the base
model was not trained on): in-sample scores are over-confident and would make every
stage look sharper than it really is. Multiplier 1 everywhere = the base model.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .base import (
    DEFAULT_STAGES, N_SIMS, fit_stage_multipliers, prediction_frame,
    simulate_positions_staged,
)

FOLDS = 3


class StagewiseModel:
    """A base score model + one Plackett-Luce temperature per stage of the order."""

    def __init__(self, base: Callable[[], Any], name: str,
                 stages: tuple[int, ...] = DEFAULT_STAGES, folds: int = FOLDS,
                 monotone: bool = False, n_sims: int = N_SIMS, seed: int = 0) -> None:
        self.base = base
        self.name = name
        self.stages = tuple(stages)
        self.folds = folds
        self.monotone = monotone
        self.n_sims = n_sims
        self.seed = seed

    def params(self) -> dict[str, Any]:
        out: dict[str, Any] = {"stages": list(self.stages), "folds": self.folds,
                               "monotone": self.monotone, "n_sims": self.n_sims,
                               "seed": self.seed}
        if hasattr(self, "multipliers_"):
            out["multipliers"] = [round(float(m), 4) for m in self.multipliers_]
        if hasattr(self, "model_") and hasattr(self.model_, "params"):
            out["base"] = self.model_.params()
        return out

    def fit(self, train: pd.DataFrame) -> StagewiseModel:
        """Fit the base model on everything; fit the stage multipliers out of fold."""
        self.model_ = self.base().fit(train)
        scores, races, finish = self._out_of_fold(train)
        self.multipliers_ = fit_stage_multipliers(scores, races, finish, self.stages,
                                                  self.monotone)
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        s = self.model_.race_scores(race)
        rng = np.random.default_rng([self.seed, int(race["RaceId"].iloc[0])])
        spread = None if s.spread is None else s.scale * np.asarray(s.spread, float)
        positions = simulate_positions_staged(s.scale * np.asarray(s.scores, float), s.p_dnf,
                                              self.n_sims, rng, self.multipliers_,
                                              self.stages, spread)
        return prediction_frame(race, positions, s.p_dnf)

    def race_scores(self, race: pd.DataFrame) -> Any:
        """The base model's scores (so a stagewise model can itself be wrapped/pooled)."""
        return self.model_.race_scores(race)

    def _out_of_fold(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scaled scores for every training race from a model that did not see it.

        Races are dealt into folds in time order (every k-th race), so each fold spans
        all seasons. With fewer than 2 * folds races the in-sample scores are used
        instead (a limitation for tiny data: the multipliers then come out too sharp).
        """
        race_ids = (train.drop_duplicates("RaceId").sort_values(["Date", "RaceId"])["RaceId"]
                    .to_numpy())
        out_scores, out_races, out_finish = [], [], []
        if len(race_ids) < 2 * self.folds:
            groups = [(train, race_ids)]
        else:
            groups = []
            for k in range(self.folds):
                held = race_ids[k:: self.folds]
                groups.append((train[~train["RaceId"].isin(held)], held))
        for fit_rows, held in groups:
            model = self.model_ if fit_rows is train else self.base().fit(fit_rows)
            for race_id in held:
                race = train[train["RaceId"] == race_id]
                s = model.race_scores(race)
                out_scores.append(s.scale * np.asarray(s.scores, float))
                out_races.append(np.full(len(race), race_id))
                out_finish.append(race["Finish"].to_numpy(float))
        return (np.concatenate(out_scores), np.concatenate(out_races),
                np.concatenate(out_finish))
