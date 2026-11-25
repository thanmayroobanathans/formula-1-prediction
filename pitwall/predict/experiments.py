"""Experiment log v2 (Phase 15): one row per experiment, plus its per-race losses.

reports/experiments.csv keeps every Phase 8-14 column and adds:
  experiment_id   short content hash: same experiment on the same data = same id
  family          model family ('B0', 'B1', 'M1', 'pool', ...): trials are counted per family
  params_hash     hash of the CONFIGURATION (fitted values such as Kalman variances left out)
  phase           the project phase that ran it ('<=14' for rows written before Phase 15)
  validation      'walk_forward' or 'cpcv'
  embargo         CPCV embargo in race weekends ('' for walk-forward)
  cpcv_blocks / cpcv_test_blocks   N and k of the CPCV run ('' for walk-forward)
  data_hash       fingerprint of the dataset rows used (compare like with like)

Per-race losses go to reports/experiments/<experiment_id>.csv (small text, committed),
so a later phase can recompute bootstraps, deflated scores and PBO without retraining.
Old rows stay readable: read_log() fills the new columns for them.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

PHASE = "16"
OLD_PHASE = "<=14"
V2_COLUMNS = ["experiment_id", "family", "params_hash", "phase", "validation", "embargo",
              "cpcv_blocks", "cpcv_test_blocks", "data_hash"]
VALIDATIONS = ("walk_forward", "cpcv")
FITTED_KEYS = frozenset({"strength", "sim", "wet"})  # learned from data, not chosen by us
LOSS_COLUMNS = ["Split", "Block", "RaceId", "Date", "Year", "n_rows",
                "win", "podium", "points", "dnf", "winner"]
LOSS_DECIMALS = 6
# Phase 16: stagewise Plackett-Luce is its own family (its own trials in the deflated
# score, PBO and BH), whichever base model it wraps. Checked before the plain prefixes.
FAMILY_PREFIXES = [("B1 stagewise", "stagewise PL"), ("M1 stagewise", "stagewise PL"),
                   ("M4 stagewise", "stagewise PL"), ("M4", "M4"),
                   ("POOL", "pool"), ("ENS", "ensemble"), ("B0", "B0"), ("B1", "B1"),
                   ("B2", "B2"), ("M1", "M1"), ("M2", "M2"), ("M3", "M3")]


def is_validated_phase(phase: pd.Series) -> pd.Series:
    """True for runs logged by the Phase 15+ validation (phase '15', '16', ...)."""
    return pd.to_numeric(phase, errors="coerce").fillna(0) >= 15


# ---------- names and hashes ----------
def model_family(name: str) -> str:
    """The family a model belongs to: 'M1-pure strength' -> 'M1', 'POOL-log B0+B1' -> 'pool'."""
    for prefix, family in FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family
    return name.split()[0] if name.strip() else "unknown"


def config_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """The settings WE chose (fitted values removed): what makes two trials different."""
    return {k: v for k, v in params.items() if k not in FITTED_KEYS}


def params_hash(params: Mapping[str, Any] | str) -> str:
    """10-character hash of a configuration (a dict, or its JSON text from the log)."""
    if isinstance(params, str):
        text = params
        try:
            params = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            return _short(text)
    return _short(json.dumps(config_params(params), sort_keys=True, default=str))


def data_hash(rows: pd.DataFrame) -> str:
    """12-character fingerprint of the dataset rows an experiment used."""
    if rows.empty:
        return "empty"
    hashed = pd.util.hash_pandas_object(rows.reset_index(drop=True), index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()[:12]


def experiment_id(**spec: Any) -> str:
    """A content hash of everything that defines an experiment (12 characters)."""
    return _short(json.dumps(spec, sort_keys=True, default=str), 12)


def _short(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


# ---------- the CSV log ----------
def read_log(path: Path | str) -> pd.DataFrame:
    """The whole log as a table, v2 columns filled for old rows. Empty if no file."""
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=V2_COLUMNS)
    log = pd.read_csv(target, dtype=str, keep_default_na=False)
    return fill_v2(log)


def fill_v2(log: pd.DataFrame) -> pd.DataFrame:
    """Add or complete the v2 columns of a log table (old rows get derived values)."""
    out = log.copy()
    for column in V2_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out = out.astype({c: str for c in V2_COLUMNS})
    missing = out["experiment_id"] == ""
    if missing.any():
        out.loc[missing, "experiment_id"] = [
            experiment_id(**{k: row.get(k, "") for k in ("timestamp", "git", "model", "params",
                                                         "moment", "seasons", "retrain_every")})
            for row in out.loc[missing].to_dict("records")]
    for column, fill in (("family", lambda r: model_family(str(r.get("model", "")))),
                         ("params_hash", lambda r: params_hash(str(r.get("params", "")))),
                         ("phase", lambda r: OLD_PHASE),
                         ("validation", lambda r: "walk_forward")):
        empty = out[column] == ""
        if empty.any():
            out.loc[empty, column] = [fill(r) for r in out.loc[empty].to_dict("records")]
    return out


def upgrade_log(path: Path | str, columns: list[str]) -> None:
    """Rewrite an old-format log with the v2 header (old values kept, new ones derived)."""
    target = Path(path)
    if not target.exists():
        return
    with target.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if all(c in header for c in columns):
        return
    log = fill_v2(pd.read_csv(target, dtype=str, keep_default_na=False))
    for column in columns:
        if column not in log.columns:
            log[column] = ""
    extra = [c for c in log.columns if c not in columns]
    log[columns + extra].to_csv(target, index=False)


def append_rows(path: Path | str, rows: Iterable[Mapping[str, Any]],
                columns: list[str]) -> int:
    """Append rows (upgrading an old file first). Rows whose experiment_id is already in
    the log are skipped: re-running an identical experiment adds nothing. Returns the
    number of rows written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    upgrade_log(target, columns)
    known = set(read_log(target)["experiment_id"]) if target.exists() else set()
    new_file = not target.exists()
    written = 0
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            if row["experiment_id"] in known:
                continue
            writer.writerow({c: row.get(c, "") for c in columns})
            known.add(row["experiment_id"])
            written += 1
    return written


# ---------- per-race losses ----------
def losses_dir_for(log_path: Path | str) -> Path:
    """reports/experiments.csv -> reports/experiments/ (next to the log)."""
    path = Path(log_path)
    return path.parent / path.stem


def write_race_losses(directory: Path | str, exp_id: str, table: pd.DataFrame) -> Path:
    """Save one experiment's per-race losses as reports/experiments/<id>.csv."""
    target = Path(directory) / f"{exp_id}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    for column in LOSS_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    if "Date" in out and pd.api.types.is_datetime64_any_dtype(out["Date"]):
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out[LOSS_COLUMNS].to_csv(target, index=False, float_format=f"%.{LOSS_DECIMALS}g")
    return target


def read_race_losses(directory: Path | str, exp_id: str) -> pd.DataFrame:
    """Per-race losses of one experiment (FileNotFoundError if it was never saved)."""
    target = Path(directory) / f"{exp_id}.csv"
    if not target.exists():
        raise FileNotFoundError(f"No per-race losses for experiment {exp_id} at {target}.")
    table = pd.read_csv(target)
    table["Date"] = pd.to_datetime(table["Date"], errors="coerce")
    return table


# ---------- trials ----------
def count_trials(log: pd.DataFrame, family: str, moment: str) -> int:
    """How many DIFFERENT configurations of a family were tried at a moment (any
    validation, any data): the 'number of trials' for the deflated score."""
    if log.empty:
        return 0
    rows = log[(log["family"] == family) & (log["moment"] == moment)]
    return int(rows["params_hash"].nunique())


def trials_table(log: pd.DataFrame) -> pd.DataFrame:
    """Distinct configurations per (family, moment)."""
    if log.empty:
        return pd.DataFrame(columns=["family", "moment", "trials"])
    return (log.groupby(["family", "moment"])["params_hash"].nunique()
            .rename("trials").reset_index())


def find_experiment(log: pd.DataFrame, model: str, moment: str, validation: str,
                    embargo: int | None = None, data: str | None = None,
                    season: int | None = None) -> pd.Series | None:
    """The LATEST logged experiment matching the filters (None if there is none)."""
    if log.empty:
        return None
    rows = log[(log["model"] == model) & (log["moment"] == moment)
               & (log["validation"] == validation)]
    if embargo is not None:
        rows = rows[rows["embargo"] == str(embargo)]
    if data is not None:
        rows = rows[rows["data_hash"] == data]
    if season is not None:
        rows = rows[rows["seasons"].str.split().apply(lambda s: str(season) in s)]
    if rows.empty:
        return None
    return rows.iloc[-1]


# ---------- merging logs from parallel jobs ----------
SIDE_FILES = ("validation_front_of_grid.csv", "validation_temperatures.csv")
TIMES_FILE = "validation_times.csv"


def merge_logs(sources: Iterable[Path | str], target: Path | str,
               columns: list[str]) -> int:
    """Merge experiment logs written by parallel jobs (e.g. one per moment) into `target`:
    rows appended (known ids skipped), per-race loss files and the validation side files
    copied next to it, run times concatenated. Returns the number of new rows."""
    import shutil

    target = Path(target)
    written = 0
    times = []
    for source in (Path(s) for s in sources):
        if not source.exists():
            continue
        log = read_log(source)
        written += append_rows(target, log.to_dict("records"), columns)
        losses = losses_dir_for(source)
        if losses.exists():
            out = losses_dir_for(target)
            out.mkdir(parents=True, exist_ok=True)
            for file in losses.glob("*.csv"):
                shutil.copy2(file, out / file.name)
        for name in SIDE_FILES:
            if (source.parent / name).exists():
                shutil.copy2(source.parent / name, target.parent / name)
        if (source.parent / TIMES_FILE).exists():
            times.append(pd.read_csv(source.parent / TIMES_FILE))
    if times:
        pd.concat(times, ignore_index=True).to_csv(target.parent / TIMES_FILE, index=False)
    return written
