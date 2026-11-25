"""The shared model interface, and the maths that turns scores into race probabilities.

A model only needs one number per driver: a 'score' (higher = faster). Plackett-Luce
turns scores into a finishing order: the winner is picked with probability proportional
to exp(score), then 2nd from the rest, and so on. The Gumbel trick does this in one
step: add random Gumbel noise to every score and sort. Repeat 20,000 times, count
where each driver lands, and you get a drivers x positions probability matrix.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import numpy as np
import pandas as pd
from scipy import optimize

from ..dataset import feature_columns

PODIUM, POINTS = 3, 10
N_SIMS = 20_000
SMOOTHING = 0.5               # pseudo-count per (driver, position) cell
TOLERANCE = 1e-6
SCALE_BOUNDS = (1e-3, 50.0)   # allowed temperature (score scale) values
POSITION_PREFIX = "P_pos_"
PROB_COLUMNS = ["P_win", "P_podium", "P_points", "P_dnf"]
OUTPUT_COLUMNS = ["RaceId", "DriverId", "Driver", *PROB_COLUMNS, "ExpectedPosition"]
FitAndScore = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


class RaceModel(Protocol):
    """What every model provides. predict() gets ONE race of ONE moment, without labels."""

    name: str

    def fit(self, train: pd.DataFrame) -> RaceModel:
        """Learn from dataset rows of one moment (labels included); returns self."""
        ...

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """OUTPUT_COLUMNS plus P_pos_1..P_pos_n, one row per driver."""
        ...

    def params(self) -> dict[str, Any]:
        """Settings worth logging with each experiment."""
        ...


# ---------- rows -> numbers ----------
def moment_of(rows: pd.DataFrame) -> str:
    """The single moment ('pre', 'practice' or 'quali') these rows belong to."""
    moments = rows["Moment"].unique()
    if len(moments) != 1:
        raise ValueError(f"Rows must come from exactly one moment, got {list(moments)}.")
    return str(moments[0])


def features_of(rows: pd.DataFrame) -> list[str]:
    """Feature columns allowed at these rows' moment."""
    return feature_columns(moment_of(rows))


Z_CLIP: float | None = 5.0  # standardised inputs are clipped to +-Z_CLIP (None = off)


def design_matrix(rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Feature values as a float array (leftover NaN and +-inf become 0, never a huge
    number: np.nan_to_num would otherwise turn inf into 1.8e308)."""
    return np.nan_to_num(rows[features].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)


def standardise(scaler: Any, raw: np.ndarray) -> np.ndarray:
    """scaler.transform, then clipped to +-Z_CLIP standard deviations: one extreme input
    (a wet-qualifying gap, a pit-lane start) cannot push a linear model to 0% or 100%."""
    z = np.asarray(scaler.transform(raw), float)
    return z if Z_CLIP is None else np.clip(z, -Z_CLIP, Z_CLIP)


# ---------- scores -> probabilities ----------
def simulate_positions(scores: np.ndarray, p_dnf: np.ndarray, n_sims: int,
                       rng: np.random.Generator,
                       score_sd: np.ndarray | None = None) -> np.ndarray:
    """Plackett-Luce Monte Carlo: matrix[i, k] = P(driver i finishes in position k+1).

    Drivers who retire in a simulation go to the back (in random order). Every
    simulated race is a full ordering, so each row and each column sums to 1.
    score_sd (optional): each simulation first draws every driver's score from
    N(score, score_sd^2), i.e. "we are not sure how fast he really is". None keeps
    the Phase 8 behaviour (and random numbers) exactly.
    """
    n = len(scores)
    keys = scores[None, :] + rng.gumbel(size=(n_sims, n))
    retired = rng.random((n_sims, n)) < p_dnf[None, :]
    if score_sd is not None:  # drawn last, so the other draws match Phase 8
        keys = keys + np.asarray(score_sd, float)[None, :] * rng.standard_normal((n_sims, n))
    keys = np.where(retired, keys - 1e9, keys)
    order = np.argsort(-keys, axis=1)                  # order[s, k] = driver in place k+1
    place = np.empty_like(order)
    np.put_along_axis(place, order, np.arange(n)[None, :], axis=1)  # place[s, i] = slot
    cells = np.arange(n)[None, :] * n + place
    counts = np.bincount(cells.ravel(), minlength=n * n).reshape(n, n)
    return (counts + SMOOTHING) / (n_sims + SMOOTHING * n)


# ---------- stagewise Plackett-Luce (Phase 16) ----------
DEFAULT_STAGES = (1, 3, 10)        # stage ends: 1st | 2nd-3rd | 4th-10th | 11th onwards
MULTIPLIER_BOUNDS = (0.05, 20.0)   # allowed stage multipliers of the model's temperature


class RaceScores(NamedTuple):
    """What a score model says about one race, before turning it into probabilities."""

    scores: np.ndarray            # higher = faster (raw: the temperature is separate)
    spread: np.ndarray | None     # uncertainty of each score (raw units), or None
    p_dnf: np.ndarray
    scale: float                  # the model's own (single) temperature


def stage_index(n_places: int, stages: tuple[int, ...] = DEFAULT_STAGES) -> np.ndarray:
    """stage_index[k] = which stage place k+1 belongs to (0 = the first stage)."""
    places = np.arange(1, n_places + 1)
    return np.searchsorted(np.asarray(stages), places, side="left")


def simulate_positions_staged(scores: np.ndarray, p_dnf: np.ndarray, n_sims: int,
                              rng: np.random.Generator, multipliers: np.ndarray,
                              stages: tuple[int, ...] = DEFAULT_STAGES,
                              score_sd: np.ndarray | None = None) -> np.ndarray:
    """Stagewise Plackett-Luce Monte Carlo (Phase 16).

    Ordinary Plackett-Luce uses one temperature for every place. Here each STAGE of
    the order (by default 1st | 2nd-3rd | 4th-10th | 11th+) has its own: the places of
    stage s are filled, best first, from softmax(multiplier_s * score) over the drivers
    not placed yet (Gumbel trick: the top-m keys of multiplier * score + Gumbel noise
    are a Plackett-Luce draw of m places). Retired drivers go to the back.
    """
    n = len(scores)
    strength = np.asarray(scores, float)[None, :]
    if score_sd is not None:
        strength = strength + np.asarray(score_sd, float)[None, :] * rng.standard_normal(
            (n_sims, n))
    retired = rng.random((n_sims, n)) < np.asarray(p_dnf, float)[None, :]
    placed = np.zeros((n_sims, n), dtype=bool)
    order = np.empty((n_sims, n), dtype=np.int64)
    rows = np.arange(n_sims)[:, None]
    ends = [min(e, n) for e in stages if e < n] + [n]
    start = 0
    for s, end in enumerate(ends):
        if end <= start:
            continue
        m = float(multipliers[min(s, len(multipliers) - 1)])
        keys = m * strength + rng.gumbel(size=(n_sims, n))
        keys = np.where(retired, keys - 1e9, keys)   # retired: only once finishers run out
        keys = np.where(placed, -np.inf, keys)
        take = np.argsort(-keys, axis=1)[:, : end - start]
        order[:, start:end] = take
        placed[rows, take] = True
        start = end
    place = np.empty_like(order)
    np.put_along_axis(place, order, np.arange(n)[None, :], axis=1)
    cells = np.arange(n)[None, :] * n + place
    counts = np.bincount(cells.ravel(), minlength=n * n).reshape(n, n)
    return (counts + SMOOTHING) / (n_sims + SMOOTHING * n)


def staged_pl_loglik(u: np.ndarray, mask: np.ndarray, multipliers: np.ndarray,
                     stages: tuple[int, ...] = DEFAULT_STAGES) -> float:
    """Plackett-Luce log-likelihood with one temperature multiplier per stage.

    u[r, k] = score of the driver who finished k-th in race r (finishers only). Place k
    in stage s adds  m_s * u_k - log(sum_{j >= k} exp(m_s * u_j)).
    """
    if mask.size == 0:
        return 0.0
    stage = stage_index(u.shape[1], stages)
    total = 0.0
    for s in np.unique(stage):
        m = float(multipliers[min(int(s), len(multipliers) - 1)])
        v = np.where(mask, m * u, -np.inf)
        top = np.max(np.where(mask, v, -np.inf), axis=1, keepdims=True)
        top = np.where(np.isfinite(top), top, 0.0)
        e = np.where(mask, np.exp(v - top), 0.0)
        denom = np.flip(np.cumsum(np.flip(e, axis=1), axis=1), axis=1)
        here = mask & (stage[None, :] == s)
        safe = np.where(here, denom, 1.0)
        total += float(np.sum(np.where(here, v - top - np.log(safe), 0.0)))
    return total


def fit_stage_multipliers(scores: np.ndarray, races: np.ndarray, finish: np.ndarray,
                          stages: tuple[int, ...] = DEFAULT_STAGES,
                          monotone: bool = False) -> np.ndarray:
    """Stage multipliers maximising the stagewise PL likelihood of the observed orders.

    scores must be OUT-OF-SAMPLE (scores of races the model was not trained on) and
    already multiplied by the model's own temperature: then multiplier 1 everywhere
    means "the model's single temperature was right". monotone=True forces
    m_1 >= m_2 >= ... (the front is never less predictable than the back).
    """
    tensor, mask = finish_tensor(scores, races, finish)
    n_stages = len(stages) + 1
    if not mask.any():
        return np.ones(n_stages)
    u = tensor[..., 0]
    low, high = np.log(MULTIPLIER_BOUNDS[0]), np.log(MULTIPLIER_BOUNDS[1])

    def to_multipliers(x: np.ndarray) -> np.ndarray:
        if not monotone:
            return np.exp(x)
        # x[0] = log of the last stage; the others are log-increments >= 0 (softplus)
        steps = np.concatenate([[x[0]], np.logaddexp(0.0, x[1:])])
        return np.exp(np.cumsum(steps))[::-1]

    def loss(x: np.ndarray) -> float:
        return -staged_pl_loglik(u, mask, to_multipliers(x), stages)

    start = np.zeros(n_stages) if not monotone else np.full(n_stages, -3.0)
    if monotone:
        start[0] = 0.0
    bounds = [(low, high)] * n_stages if not monotone else [(low, high)] + [(-10, 5)] * (
        n_stages - 1)
    result = optimize.minimize(loss, start, method="L-BFGS-B", bounds=bounds)
    return np.clip(to_multipliers(result.x), *MULTIPLIER_BOUNDS)


def prediction_frame(race: pd.DataFrame, positions: np.ndarray,
                     p_dnf: np.ndarray) -> pd.DataFrame:
    """The standard output table from a drivers x positions matrix."""
    n = len(race)
    ids = race[["RaceId", "DriverId", "Driver"]].reset_index(drop=True)
    summary = pd.DataFrame({
        "P_win": positions[:, 0],
        "P_podium": positions[:, :PODIUM].sum(axis=1),
        "P_points": positions[:, :POINTS].sum(axis=1),
        "P_dnf": np.clip(np.asarray(p_dnf, float), 0.0, 1.0),
        "ExpectedPosition": positions @ np.arange(1, n + 1),
    })
    grid = pd.DataFrame(positions, columns=[f"{POSITION_PREFIX}{k}" for k in range(1, n + 1)])
    return pd.concat([ids, summary, grid], axis=1)


def predict_from_scores(race: pd.DataFrame, scores: np.ndarray, p_dnf: np.ndarray,
                        scale: float = 1.0, n_sims: int = N_SIMS, seed: int = 0,
                        score_sd: np.ndarray | None = None) -> pd.DataFrame:
    """Scores (higher = faster) and DNF chances -> the standard prediction table.

    The random generator is seeded by (seed, RaceId): the same race always gives the
    same numbers. score_sd is in score units BEFORE scaling (it is scaled too).
    """
    rng = np.random.default_rng([seed, int(race["RaceId"].iloc[0])])
    strength = scale * np.asarray(scores, float)
    spread = None if score_sd is None else scale * np.asarray(score_sd, float)
    chances = np.asarray(p_dnf, float)
    positions = simulate_positions(strength, chances, n_sims, rng, spread)
    return prediction_frame(race, positions, chances)


def sinkhorn(matrix: np.ndarray, iters: int = 1000, tol: float = 1e-12) -> np.ndarray:
    """Rescale rows and columns in turn until every row and column sums to 1.

    Turns 'P(position | slot)' rows into a consistent race: each driver gets exactly one
    position and each position exactly one driver. All entries must be > 0.
    """
    out = np.asarray(matrix, float).copy()
    for _ in range(iters):
        out /= out.sum(axis=1, keepdims=True)
        out /= out.sum(axis=0, keepdims=True)
        if np.abs(out.sum(axis=1) - 1).max() < tol:
            break
    return out


def position_matrix(pred: pd.DataFrame) -> np.ndarray:
    """The P_pos_k columns as an array (missing positions, e.g. smaller fields, = 0)."""
    columns = sorted((c for c in pred.columns if c.startswith(POSITION_PREFIX)),
                     key=lambda c: int(c.removeprefix(POSITION_PREFIX)))
    return np.nan_to_num(pred[columns].to_numpy(float), nan=0.0)


def check_consistency(pred: pd.DataFrame, tol: float = TOLERANCE) -> None:
    """Raise ValueError unless every probability is in [0, 1] and, per race,
    sum(P_win) = 1, sum(P_podium) = min(3, starters), sum(P_points) = min(10, starters)."""
    probs = pred[[c for c in pred.columns if c.startswith("P_")]].to_numpy(float)
    probs = probs[~np.isnan(probs)]
    if ((probs < -tol) | (probs > 1 + tol)).any():
        raise ValueError("A probability is outside [0, 1].")
    for race, group in pred.groupby("RaceId"):
        n = len(group)
        for column, wanted in (("P_win", 1), ("P_podium", min(PODIUM, n)),
                               ("P_points", min(POINTS, n))):
            got = float(group[column].sum())
            if abs(got - wanted) > tol:
                raise ValueError(f"Race {race}: sum of {column} is {got:.6f}, not {wanted}.")


# ---------- fitting the score scale (temperature) ----------
def finish_tensor(values: np.ndarray, races: np.ndarray,
                  finish: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group rows by race (finishers only, in finishing order), padded to (races, n, d).

    mask[r, k] is True where a real finisher sits. Races with < 2 finishers are skipped.
    DNFs are left out: they are handled by the separate DNF models.
    """
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    races, finish = np.asarray(races), np.asarray(finish, float)
    groups = []
    for race in np.unique(races):
        idx = np.flatnonzero((races == race) & ~np.isnan(finish))
        if len(idx) >= 2:
            groups.append(idx[np.argsort(finish[idx], kind="stable")])
    width = max((len(g) for g in groups), default=0)
    tensor = np.zeros((len(groups), width, values.shape[1]))
    mask = np.zeros((len(groups), width), dtype=bool)
    for r, idx in enumerate(groups):
        tensor[r, :len(idx)] = values[idx]
        mask[r, :len(idx)] = True
    return tensor, mask


def pl_loglik_and_grad(u: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Plackett-Luce log-likelihood of the observed orders, and its gradient w.r.t. u.

    u[r, k] is the strength of the driver who finished k-th in race r. Each place adds
    log P(that driver beats everyone still left) = u_k - log(sum over j >= k of exp(u_j)).
    """
    if mask.size == 0:
        return 0.0, np.zeros_like(u)
    u = np.where(mask, u, -np.inf)
    top = np.max(u, axis=1, keepdims=True)
    e = np.exp(u - top)                                   # 0 where there is no driver
    denom = np.flip(np.cumsum(np.flip(e, axis=1), axis=1), axis=1)
    safe = np.where(mask, denom, 1.0)
    loglik = float(np.sum(np.where(mask, u - top - np.log(safe), 0.0)))
    inverse = np.where(mask, 1.0 / safe, 0.0)
    grad = np.where(mask, 1.0 - e * np.cumsum(inverse, axis=1), 0.0)
    return loglik, grad


def fit_scale(scores: np.ndarray, races: np.ndarray, finish: np.ndarray) -> float:
    """The temperature beta that makes Plackett-Luce(beta * score) fit the results best."""
    tensor, mask = finish_tensor(scores, races, finish)
    if not mask.any():
        return 1.0
    values = tensor[..., 0]

    def loss(beta: float) -> float:
        return -pl_loglik_and_grad(beta * values, mask)[0]

    result = optimize.minimize_scalar(loss, bounds=SCALE_BOUNDS, method="bounded")
    return float(result.x)


def out_of_fold_scale(train: pd.DataFrame, fit_and_score: FitAndScore,
                      folds: int = 3, seed: int = 0) -> float:
    """Temperature fitted on scores for races the model was NOT trained on.

    fit_and_score(fit_rows, score_rows) trains a fresh model on fit_rows and returns
    scores for score_rows. In-sample scores look over-confident (the model has already
    seen those races), which would make the fitted temperature too sharp.
    """
    races = np.array(sorted(train["RaceId"].unique()))
    if len(races) < 2 * folds:
        scores = fit_and_score(train, train)   # too little data to hold races out
    else:
        rng = np.random.default_rng(seed)
        fold_of = dict(zip(rng.permutation(races), np.arange(len(races)) % folds))
        fold = train["RaceId"].map(fold_of).to_numpy()
        scores = np.empty(len(train))
        for k in range(folds):
            held = fold == k
            scores[held] = fit_and_score(train[~held], train[held])
    return fit_scale(scores, train["RaceId"].to_numpy(), train["Finish"].to_numpy(float))


# ---------- DNF classifiers ----------
class ConstantRate:
    """Fallback when the training labels have only one class (e.g. no DNFs at all)."""

    def __init__(self, labels: np.ndarray) -> None:
        self.rate = (float(np.sum(labels)) + 1.0) / (len(labels) + 2.0)  # Laplace

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Same two-column shape as scikit-learn: [P(no), P(yes)]."""
        p = np.full(len(features), self.rate)
        return np.column_stack([1 - p, p])


def fit_binary(model: Any, features: np.ndarray, labels: np.ndarray) -> Any:
    """Fit a classifier, or fall back to a smoothed constant rate if only one class."""
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return ConstantRate(labels)
    return model.fit(features, labels)
