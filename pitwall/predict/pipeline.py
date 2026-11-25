"""The final pipeline for Phase 12's CLI and the single 2026 run.

FinalPipeline(config) fits the chosen models on ALL rows of the moment dated before a
race, combines them with ensemble weights fitted on earlier backtest predictions
(`history`), optionally calibrates, and predicts that one race. 2026 races need
final=True. The config lives in reports/final_config.json.
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..errors import DataNotAvailableError
from .backtest import REPORTS_DIR, check_lock, check_no_future, git_hash, strip_outcomes
from .calibration import apply_platt, fit_platt, fit_temperature, temper
from .ensemble import L2_TO_EQUAL, MIN_TRAIN_RACES, build_stacks, fit_weights, pool
from .models import (
    BayesianModel, GridSlotBaseline, LightGBMRanker, LogisticModel, NeuralRanker,
    RaceSimulator, StagewiseModel, StrengthModel,
)
from .models.base import check_consistency, position_matrix, prediction_frame

CONFIG_PATH = REPORTS_DIR / "final_config.json"


@dataclass
class MomentChoice:
    """What to run at one moment."""

    models: list[str]
    method: str = "single"     # 'single', 'equal', 'linear' or 'log'
    calibrate: bool = False
    l2: float = L2_TO_EQUAL


@dataclass
class PipelineConfig:
    """The chosen pipeline per moment (saved as JSON)."""

    moments: dict[str, MomentChoice]
    n_seeds: int = 10
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | str = CONFIG_PATH) -> Path:
        """Write the config (with a timestamp and git hash)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {"moments": {m: asdict(c) for m, c in self.moments.items()},
                "n_seeds": self.n_seeds, "note": self.note, "extra": self.extra,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "git": git_hash()}
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> PipelineConfig:
        """Read a saved config."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        moments = {m: MomentChoice(**c) for m, c in data["moments"].items()}
        return cls(moments, data.get("n_seeds", 10), data.get("note", ""),
                   data.get("extra", {}))


def default_config() -> PipelineConfig:
    """Used until a backtest has written reports/final_config.json: an equal-weight blend
    of B1 (logistic) and M1 (Kalman strength), both fast on a CPU."""
    choice = {m: MomentChoice(["B1 logistic", "M1 strength"], method="equal")
              for m in ("pre", "practice", "quali")}
    return PipelineConfig(choice, n_seeds=10,
                          note="default: no backtest has chosen a pipeline yet")


def load_config(path: Path | str = CONFIG_PATH) -> PipelineConfig:
    """The saved config if it exists, else default_config()."""
    return PipelineConfig.load(path) if Path(path).exists() else default_config()


def model_factories(n_seeds: int) -> dict[str, Callable[[], Any]]:
    """Model name (as in the backtest) -> a fresh, unfitted model."""
    return {"B0 grid slot": GridSlotBaseline, "B1 logistic": LogisticModel,
            "B2 LightGBM": LightGBMRanker, "M1 strength": StrengthModel,
            "M1-pure strength": lambda: StrengthModel(pure=True),
            "M2-full": lambda: NeuralRanker(n_seeds=n_seeds), "M3 simulator": RaceSimulator,
            # Phase 16: one Plackett-Luce temperature per stage of the order
            "B1 stagewise": lambda: StagewiseModel(LogisticModel, "B1 stagewise"),
            "M1 stagewise": lambda: StagewiseModel(StrengthModel, "M1 stagewise"),
            # Phase 16: the hierarchical Bayesian car/driver model (needs the bayes extra)
            "M4 Bayesian": BayesianModel,
            "M4 stagewise": lambda: StagewiseModel(BayesianModel, "M4 stagewise")}


class FinalPipeline:
    """Fit-and-predict for one race at one moment, using only information before it."""

    def __init__(self, config: PipelineConfig, history: pd.DataFrame | None = None) -> None:
        self.config = config
        self.history = history  # backtest predictions with labels, for weights/calibration

    def predict(self, data: pd.DataFrame, race_id: int, moment: str,
                final: bool = False, live: bool = False) -> pd.DataFrame:
        """The prediction table for one race. 2026 (locked) needs final=True.

        live=True is for a race that has NOT been run (no labels at all): forecasting it
        is not an evaluation on the locked season, so the lock does not apply.
        """
        if moment not in self.config.moments:
            raise ValueError(f"No pipeline configured for moment '{moment}'.")
        rows = data[data["Moment"] == moment]
        race = rows[rows["RaceId"] == race_id]
        if race.empty:
            raise DataNotAvailableError(f"Race {race_id} has no '{moment}' rows.")
        date = pd.Timestamp(race["Date"].iloc[0])
        if live and "Finish" in race and race["Finish"].notna().any():
            raise ValueError("live=True is only for races without results.")
        check_lock([int(race["Year"].iloc[0])], final or live)
        train = rows[rows["Date"] < date]
        check_no_future(train, date, moment)
        choice = self.config.moments[moment]
        factories = model_factories(self.config.n_seeds)
        unknown = [m for m in choice.models if m not in factories]
        if unknown:
            raise ValueError(f"Unknown models {unknown}. Known: {', '.join(factories)}.")
        visible = strip_outcomes(race)
        preds = [factories[m]().fit(train).predict(visible) for m in choice.models]
        if len(preds) == 1 or choice.method == "single":
            out = preds[0]
        else:
            out = self._combine(preds, visible, moment, date, choice)
        check_consistency(out)
        return out

    def _combine(self, preds: list[pd.DataFrame], race: pd.DataFrame, moment: str,
                 date: pd.Timestamp, choice: MomentChoice) -> pd.DataFrame:
        n = len(race)
        matrices = np.stack([position_matrix(p)[:, :n] for p in preds])
        dnf = np.stack([p["P_dnf"].to_numpy(float) for p in preds])
        past = ([] if self.history is None else
                [s for s in build_stacks(self.history, moment, choice.models) if s.date < date])
        weights = (fit_weights(past, choice.method, choice.l2) if past
                   else np.full(len(preds), 1.0 / len(preds)))
        positions, p_dnf = pool(matrices, weights, choice.method), weights @ dnf
        if choice.calibrate and len(past) >= MIN_TRAIN_RACES:
            pooled = [pool(s.matrices, weights, choice.method) for s in past]
            positions = temper(positions, fit_temperature(pooled, [s.order for s in past]))
            a, b = fit_platt(np.concatenate([weights @ s.dnf for s in past]),
                             np.concatenate([s.frame["DNF"].to_numpy(float) for s in past]))
            p_dnf = apply_platt(p_dnf, a, b)
        return prediction_frame(race, positions, p_dnf)
