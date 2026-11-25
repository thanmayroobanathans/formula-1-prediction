"""Phase 15: combinatorial purged cross-validation (CPCV) for races.

Walk-forward (backtest.py) stays the PRIMARY, time-honest test. CPCV is the robustness
check: it asks "would the score look the same on other stretches of history?"

1. The races of 2018-2025 (never 2026) are cut, in time order, into N consecutive
   blocks (default 8). Every choice of k test blocks (default 2) is one split:
   C(8, 2) = 28 splits. Each split trains on the other blocks and tests on its k blocks.
2. Purge + embargo. Features are rolling windows over PAST races (form over 5-10 races,
   championship points), so a training race just AFTER a test block carries the test
   races' results inside its features. The `embargo` races after each test block are
   removed from training. (The purge, the training races inside a test block's date
   span, is empty for one-race labels; it is applied anyway as a guard.)
3. The splits are assembled into phi = k/N * C(N, k) = 7 full backtest PATHS (Lopez de
   Prado, Advances in Financial ML, ch. 12): every path predicts every race exactly once,
   each with a model that never trained on that race's block. Each model gets a
   distribution of 7 path scores instead of one number.

Same RaceModel interface and leakage guards as the backtest; any training race inside a
test block or its embargo raises LeakageError.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from ..errors import DataNotAvailableError
from .backtest import LeakageError, _with_labels, check_lock, strip_outcomes
from .metrics import race_loss_table, summarise, weighted_loss
from .models.base import RaceModel, check_consistency

CPCV_FIRST_YEAR, CPCV_LAST_YEAR = 2018, 2025
DEFAULT_BLOCKS, DEFAULT_TEST_BLOCKS = 8, 2
# Phase 16: 10 races, the longest look-back of any form feature (DNF rates over 10
# races), so no training race carries a test race's result inside its features.
DEFAULT_EMBARGO = 10
SENSITIVITY_EMBARGOES = (0, 1, 3)
Progress = Callable[[str], None]


@dataclass(frozen=True)
class CPCVSplit:
    """One train/test split (RaceIds)."""

    index: int
    test_blocks: tuple[int, ...]
    train: tuple[int, ...]
    test: tuple[int, ...]
    embargoed: tuple[int, ...]   # removed from training: just after a test block
    purged: tuple[int, ...]      # removed from training: inside a test block's dates


@dataclass(frozen=True)
class CPCVPlan:
    """The blocks, splits and paths of one CPCV run."""

    calendar: pd.DataFrame       # RaceId, Date, Year, Position, Block (time order)
    n_blocks: int
    n_test_blocks: int
    embargo: int
    splits: tuple[CPCVSplit, ...] = field(repr=False)

    @property
    def n_paths(self) -> int:
        """phi = k / N * C(N, k)."""
        return n_paths(self.n_blocks, self.n_test_blocks)

    def paths(self) -> list[dict[int, int]]:
        """Path p -> {block: index of the split whose prediction that block uses}."""
        return assemble_paths(self.n_blocks, self.n_test_blocks)


# ---------- planning ----------
def n_paths(n_blocks: int, n_test_blocks: int) -> int:
    """Number of full backtest paths: k / N * C(N, k)."""
    return math.comb(n_blocks, n_test_blocks) * n_test_blocks // n_blocks


def split_blocks(n_blocks: int, n_test_blocks: int) -> list[tuple[int, ...]]:
    """Every choice of test blocks, in lexicographic order (split i = entry i)."""
    if not 1 <= n_test_blocks < n_blocks:
        raise ValueError("Need 1 <= test blocks < blocks.")
    return list(combinations(range(n_blocks), n_test_blocks))


def assemble_paths(n_blocks: int, n_test_blocks: int) -> list[dict[int, int]]:
    """Path j takes, for every block b, the j-th split (in order) that tests b.

    Each block is tested in C(N-1, k-1) splits, which is exactly the number of paths,
    so every path covers all N blocks once.
    """
    tests = split_blocks(n_blocks, n_test_blocks)
    per_block = {b: [i for i, t in enumerate(tests) if b in t] for b in range(n_blocks)}
    count = n_paths(n_blocks, n_test_blocks)
    return [{b: per_block[b][j] for b in range(n_blocks)} for j in range(count)]


def race_calendar(data: pd.DataFrame, first_year: int = CPCV_FIRST_YEAR,
                  last_year: int = CPCV_LAST_YEAR) -> pd.DataFrame:
    """One row per race (RaceId, Date, Year) from first_year to last_year, time order."""
    check_lock(range(first_year, last_year + 1), final=False)  # 2026 is never used here
    races = (data[(data["Year"] >= first_year) & (data["Year"] <= last_year)]
             .drop_duplicates("RaceId")[["RaceId", "Date", "Year"]]
             .sort_values(["Date", "RaceId"]).reset_index(drop=True))
    if races.empty:
        raise DataNotAvailableError(f"No races between {first_year} and {last_year}.")
    races["Position"] = np.arange(len(races))
    return races


def plan_cpcv(data: pd.DataFrame, n_blocks: int = DEFAULT_BLOCKS,
              n_test_blocks: int = DEFAULT_TEST_BLOCKS, embargo: int = DEFAULT_EMBARGO,
              first_year: int = CPCV_FIRST_YEAR,
              last_year: int = CPCV_LAST_YEAR) -> CPCVPlan:
    """Blocks, splits (with purge and embargo) for the races of first_year..last_year."""
    if embargo < 0:
        raise ValueError("embargo must be >= 0.")
    calendar = race_calendar(data, first_year, last_year)
    if len(calendar) < 2 * n_blocks:
        raise DataNotAvailableError(
            f"{len(calendar)} races is too few for {n_blocks} blocks (need {2 * n_blocks}).")
    calendar["Block"] = 0
    for b, idx in enumerate(np.array_split(np.arange(len(calendar)), n_blocks)):
        calendar.loc[idx, "Block"] = b
    splits = tuple(_split(calendar, i, test, embargo)
                   for i, test in enumerate(split_blocks(n_blocks, n_test_blocks)))
    return CPCVPlan(calendar, n_blocks, n_test_blocks, embargo, splits)


def _split(calendar: pd.DataFrame, index: int, test_blocks: tuple[int, ...],
           embargo: int) -> CPCVSplit:
    is_test = calendar["Block"].isin(test_blocks).to_numpy()
    positions = calendar["Position"].to_numpy()
    after = np.zeros(len(calendar), dtype=bool)
    for p in positions[is_test]:
        after[(positions > p) & (positions <= p + embargo)] = True
    embargoed = after & ~is_test
    dates = calendar["Date"]
    purged = np.zeros(len(calendar), dtype=bool)
    for block in test_blocks:
        span = dates[calendar["Block"] == block]
        purged |= (dates >= span.min()).to_numpy() & (dates <= span.max()).to_numpy()
    purged &= ~is_test & ~embargoed
    train = ~is_test & ~embargoed & ~purged
    ids = calendar["RaceId"].to_numpy()
    return CPCVSplit(index, tuple(test_blocks), tuple(int(r) for r in ids[train]),
                     tuple(int(r) for r in ids[is_test]),
                     tuple(int(r) for r in ids[embargoed]), tuple(int(r) for r in ids[purged]))


def check_split(split: CPCVSplit, calendar: pd.DataFrame, embargo: int) -> None:
    """Raise LeakageError if a training race is a test race, or within `embargo` races
    after one (in calendar order), or inside a test block's date span."""
    position = dict(zip(calendar["RaceId"], calendar["Position"]))
    train, test = set(split.train), set(split.test)
    if train & test:
        raise LeakageError(f"Split {split.index}: races {sorted(train & test)[:5]} are in "
                           "both training and test.")
    test_positions = np.array([position[r] for r in test])
    for race in train:
        gap = position[race] - test_positions
        if ((gap > 0) & (gap <= embargo)).any():
            raise LeakageError(f"Split {split.index}: training race {race} is inside the "
                               f"{embargo}-race embargo after a test block.")
    dates = dict(zip(calendar["RaceId"], calendar["Date"]))
    blocks = calendar.set_index("RaceId")["Block"]
    for block in split.test_blocks:
        span = [dates[r] for r in test if blocks[r] == block]
        for race in train:
            if min(span) <= dates[race] <= max(span):
                raise LeakageError(f"Split {split.index}: training race {race} is dated "
                                   "inside a test block.")


# ---------- running ----------
@dataclass(frozen=True)
class CPCVResult:
    """Predictions (labels attached, with Split and Block columns) of every model."""

    moment: str
    plan: CPCVPlan
    predictions: pd.DataFrame
    seconds: dict[str, float]

    def models(self) -> list[str]:
        """Model names in the order they were run."""
        return list(dict.fromkeys(self.predictions["Model"]))

    def race_losses(self, model: str) -> pd.DataFrame:
        """Per (split, race) losses of one model (the table saved in the experiment log)."""
        return cpcv_race_losses(self.predictions[self.predictions["Model"] == model])

    def summary(self, model: str) -> dict[str, float]:
        """Every metric over all CPCV predictions of a model (each race counted once per
        path it appears in)."""
        return summarise(self.predictions[self.predictions["Model"] == model])

    def path_scores(self, kind: str = "win") -> pd.DataFrame:
        """One row per (model, path): the path's driver-row log loss."""
        rows = []
        for model in self.models():
            scores = path_losses(self.race_losses(model), self.plan.n_blocks,
                                 self.plan.n_test_blocks, kind)
            rows += [{"Model": model, "Path": p, "LogLoss": s} for p, s in enumerate(scores)]
        return pd.DataFrame(rows)


def run_cpcv(models: Sequence[RaceModel], data: pd.DataFrame, moment: str,
             plan: CPCVPlan, progress: Progress | None = None) -> CPCVResult:
    """Fit every model on each split's training races, predict its test races."""
    rows = data[(data["Moment"] == moment) & data["RaceId"].isin(plan.calendar["RaceId"])]
    if rows.empty:
        raise DataNotAvailableError(f"No '{moment}' rows for the CPCV races.")
    block_of = dict(zip(plan.calendar["RaceId"], plan.calendar["Block"]))
    out: list[pd.DataFrame] = []
    seconds = {m.name: 0.0 for m in models}
    for split in plan.splits:
        check_split(split, plan.calendar, plan.embargo)
        train = rows[rows["RaceId"].isin(split.train)]
        if (train["Moment"] != moment).any() or train["RaceId"].isin(split.test).any():
            raise LeakageError("Training rows from a test race or another moment.")
        for model in models:
            started = time.perf_counter()
            model.fit(train)
            for race_id in split.test:
                race = rows[rows["RaceId"] == race_id]
                prediction = model.predict(strip_outcomes(race))
                check_consistency(prediction)
                labelled = _with_labels(prediction, race, model.name)
                labelled["Split"] = split.index
                labelled["Block"] = block_of[race_id]
                out.append(labelled)
            seconds[model.name] += time.perf_counter() - started
        if progress is not None:
            progress(f"  {moment}: split {split.index + 1}/{len(plan.splits)} "
                     f"(test blocks {split.test_blocks})")
    return CPCVResult(moment, plan, pd.concat(out, ignore_index=True), seconds)


# ---------- scoring paths ----------
def cpcv_race_losses(preds: pd.DataFrame) -> pd.DataFrame:
    """Per (Split, Block, RaceId) losses from CPCV predictions of ONE model."""
    parts = []
    for (split, block), group in preds.groupby(["Split", "Block"], sort=True):
        table = race_loss_table(group)
        table.insert(0, "Block", int(block))
        table.insert(0, "Split", int(split))
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def path_losses(losses: pd.DataFrame, n_blocks: int, n_test_blocks: int,
                kind: str = "win") -> np.ndarray:
    """The driver-row log loss of each full path from a per-(split, race) loss table."""
    return np.array([weighted_loss(path_race_losses(losses, n_blocks, n_test_blocks, p), kind)
                     for p in range(n_paths(n_blocks, n_test_blocks))])


def path_race_losses(losses: pd.DataFrame, n_blocks: int, n_test_blocks: int,
                     path: int) -> pd.DataFrame:
    """The per-race losses that make up one path (one row per race). ValueError if a
    block of the path has no races (a cut or mismatched loss table)."""
    parts = []
    for block, split in assemble_paths(n_blocks, n_test_blocks)[path].items():
        part = losses[(losses["Block"] == block) & (losses["Split"] == split)]
        if part.empty:
            raise ValueError(f"CPCV path {path}: block {block} of split {split} has no "
                             "races (was the loss table cut, or N / k wrong?).")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def distribution(scores: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Mean, 5th / 50th / 95th percentile (interpolated) and count of path scores."""
    arr = np.asarray(scores, float)
    p5, p50, p95 = np.percentile(arr, [5, 50, 95])
    return {"Mean": float(arr.mean()), "P5": float(p5), "Median": float(p50),
            "P95": float(p95), "Paths": float(len(arr))}
