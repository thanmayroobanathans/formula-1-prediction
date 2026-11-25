"""Race models. Every model follows RaceModel: fit(train_rows) -> self, predict(race_rows).

Heavy libraries (scikit-learn, LightGBM, PyTorch) are imported inside fit(), so
importing this package stays light.
"""
from .bayes import BayesianModel, BayesFitError
from .base import (
    OUTPUT_COLUMNS, POSITION_PREFIX, PROB_COLUMNS, RaceModel, check_consistency, fit_scale,
    position_matrix, predict_from_scores, sinkhorn,
)
from .grid import GridSlotBaseline
from .lgbm import LightGBMRanker
from .logistic import LogisticModel
from .neural import NeuralRanker
from .scores import ScoreSource, as_score_source
from .simulator import RaceSimulator, SimParams
from .stagewise import StagewiseModel
from .strength_model import StrengthModel

__all__ = ["OUTPUT_COLUMNS", "POSITION_PREFIX", "PROB_COLUMNS", "RaceModel",
           "check_consistency", "fit_scale", "position_matrix", "predict_from_scores",
           "sinkhorn", "GridSlotBaseline", "LightGBMRanker", "LogisticModel", "NeuralRanker",
           "ScoreSource", "as_score_source", "RaceSimulator", "SimParams", "StrengthModel",
           "StagewiseModel", "BayesianModel", "BayesFitError"]
