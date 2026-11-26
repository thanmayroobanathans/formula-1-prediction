"""Phase 12: prediction records, the public track record.

Every live prediction is saved BEFORE the race as predictions/<year>-<round>-<moment>.json
and committed to git: the commit time is independent proof of when it was made. A
record is never overwritten. Once the race has run, `score_records` grades each record
and `write_track_record` keeps reports/track_record.md up to date.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from ..errors import PitwallError
from .backtest import git_hash
from .metrics import BENCHMARKS, EPS, calibration_table, row_log_loss
from .models.base import POSITION_PREFIX, PROB_COLUMNS

SCHEMA = 1
RECORDS_DIR = Path("predictions")
TRACK_RECORD_PATH = Path("reports") / "track_record.md"
DOCS_DIR = Path("docs")
BASELINE_NAME = "B0 grid slot"
SCORE_COLUMNS = ["Stem", "Year", "Round", "Event", "Moment", "Created", "Drivers",
                 "win_logloss", "podium_logloss", "points_logloss", "dnf_logloss",
                 "b0_win_logloss", "b0_podium_logloss", "Favourite", "Winner", "WinnerHit",
                 "PodiumOverlap", "P_win_of_winner"]


SURPRISE_P = 0.01   # a result the engine gave less than 1% is flagged as a surprise
UPSET_WIN_P = 0.05  # a winner the engine gave less than 5% is flagged as an upset
CHECK_COLUMNS = ["Stem", "Race", "Moment", "Kind", "Driver", "Chance", "Text"]


class RecordExistsError(PitwallError):
    """A prediction record for this race and moment already exists (never overwritten)."""


@dataclass(frozen=True)
class RecordMeta:
    """Where and when a prediction was made."""

    year: int
    round: int
    race_id: int
    event: str
    date: str
    moment: str
    pipeline: dict[str, Any]
    availability: dict[str, int]
    warnings: list[str]
    notes: list[str]


def record_stem(year: int, rnd: int, moment: str, tag: str = "") -> str:
    """'2026-23-quali', or '2026-23-quali-phase16' for an extra, labelled prediction
    (e.g. a new model added after the moment's official record was made)."""
    return f"{year}-{rnd:02d}-{moment}" + (f"-{tag}" if tag else "")


def config_hash(config: dict[str, Any]) -> str:
    """A short, stable hash of the pipeline configuration."""
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def drivers_payload(pred: pd.DataFrame, teams: dict[str, str] | None = None) -> list[dict]:
    """The standard prediction table as JSON-ready rows (P_pos as a list)."""
    pos_cols = [c for c in pred.columns if c.startswith(POSITION_PREFIX)]
    rows = []
    for row in pred.sort_values("ExpectedPosition").itertuples(index=False):
        values = row._asdict()
        rows.append({"DriverId": str(values["DriverId"]), "Driver": str(values["Driver"]),
                     "TeamId": (teams or {}).get(str(values["DriverId"]), ""),
                     **{c: round(float(values[c]), 6) for c in PROB_COLUMNS},
                     "ExpectedPosition": round(float(values["ExpectedPosition"]), 4),
                     "P_pos": [round(float(values[c]), 6) for c in pos_cols]})
    return rows


def make_record(pred: pd.DataFrame, meta: RecordMeta, baseline: pd.DataFrame | None = None,
                teams: dict[str, str] | None = None,
                now: dt.datetime | None = None) -> dict[str, Any]:
    """The JSON record of one prediction."""
    stamp = (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")
    record: dict[str, Any] = {
        "schema": SCHEMA, "year": meta.year, "round": meta.round, "race_id": meta.race_id,
        "event": meta.event, "race_date": meta.date, "moment": meta.moment,
        "created_utc": stamp, "git": git_hash(), "config_hash": config_hash(meta.pipeline),
        "version": __version__, "pipeline": meta.pipeline, "availability": meta.availability,
        "warnings": meta.warnings, "notes": meta.notes,
        "drivers": drivers_payload(pred, teams),
    }
    if baseline is not None:
        record["baseline"] = {"model": BASELINE_NAME,
                              "drivers": drivers_payload(baseline, teams)}
    return record


def save_record(record: dict[str, Any], directory: Path | str = RECORDS_DIR) -> Path:
    """Write a record; refuses (RecordExistsError) if one exists for that race and moment."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (record_stem(record["year"], record["round"], record["moment"],
                                 record.get("tag", "")) + ".json")
    try:
        with path.open("x", encoding="utf-8") as handle:  # 'x' = fail if it exists
            json.dump(record, handle, indent=1)
            handle.write("\n")
    except FileExistsError:
        raise RecordExistsError(
            f"{path} already exists. Records are never overwritten: they are the "
            "timestamped proof of what was predicted.") from None
    return path


def load_records(directory: Path | str = RECORDS_DIR) -> list[dict[str, Any]]:
    """Every record, oldest race first."""
    folder = Path(directory)
    if not folder.exists():
        return []
    records = []
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_stem"] = path.stem
        records.append(record)
    order = {"pre": 0, "practice": 1, "quali": 2}
    return sorted(records, key=lambda r: (r["year"], r["round"], order.get(r["moment"], 9)))


def record_frame(record: dict[str, Any], key: str = "drivers") -> pd.DataFrame:
    """A record's driver rows as the standard prediction table."""
    block = record[key] if key == "drivers" else record[key]["drivers"]
    frame = pd.DataFrame(block)
    if frame.empty:
        return frame
    positions = np.array(frame["P_pos"].tolist(), float)
    grid = pd.DataFrame(positions, columns=[f"{POSITION_PREFIX}{k}"
                                            for k in range(1, positions.shape[1] + 1)])
    out = pd.concat([frame.drop(columns=["P_pos"]), grid], axis=1)
    out.insert(0, "RaceId", record["race_id"])
    return out


# ---------- scoring ----------
def score_record(record: dict[str, Any], results: pd.DataFrame) -> dict[str, Any] | None:
    """Grade one record against the race's results (None if the race has none yet).

    results: one row per starter with DriverId, Finish (NaN = DNF), Win, Podium,
    PointsFinish, DNF. Drivers predicted but not starting are dropped; probabilities
    are clipped to [EPS, 1 - EPS] as in the backtest metrics.
    """
    if results.empty:
        return None
    results = results.drop(columns=["Driver"], errors="ignore")
    pred = record_frame(record).merge(results, on="DriverId", how="inner")
    if pred.empty:
        return None
    out: dict[str, Any] = {
        "Stem": record.get("_stem", record_stem(record["year"], record["round"],
                                                record["moment"])),
        "Year": record["year"], "Round": record["round"], "Event": record["event"],
        "Moment": record["moment"], "Created": record["created_utc"], "Drivers": len(pred)}
    for kind, (prob, label) in {"win": ("P_win", "Win"), "podium": ("P_podium", "Podium"),
                                "points": ("P_points", "PointsFinish"),
                                "dnf": ("P_dnf", "DNF")}.items():
        out[f"{kind}_logloss"] = float(row_log_loss(pred[prob], pred[label]).mean())
    if "baseline" in record:
        base = record_frame(record, "baseline").merge(results, on="DriverId", how="inner")
        out["b0_win_logloss"] = float(row_log_loss(base["P_win"], base["Win"]).mean())
        out["b0_podium_logloss"] = float(row_log_loss(base["P_podium"], base["Podium"]).mean())
    else:
        out["b0_win_logloss"] = out["b0_podium_logloss"] = float("nan")
    favourite = pred.loc[pred["P_win"].idxmax()]
    winners = pred[pred["Win"] == 1]
    out["Favourite"] = str(favourite["Driver"])
    out["Winner"] = str(winners["Driver"].iloc[0]) if not winners.empty else "?"
    out["WinnerHit"] = int(bool(favourite["Win"] == 1))
    top3 = set(pred.nlargest(3, "P_podium")["DriverId"])
    out["PodiumOverlap"] = len(top3 & set(pred.loc[pred["Podium"] == 1, "DriverId"])) / 3
    out["P_win_of_winner"] = (float(np.clip(winners["P_win"].iloc[0], EPS, 1 - EPS))
                              if not winners.empty else float("nan"))
    return out


def score_records(records: list[dict[str, Any]], results: pd.DataFrame) -> pd.DataFrame:
    """Scores of every record whose race has results. results needs Year and Round."""
    rows = []
    for record in records:
        race = results[(results["Year"] == record["year"]) & (results["Round"] == record["round"])]
        scored = score_record(record, race)
        if scored is not None:
            rows.append(scored)
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def check_record(record: dict[str, Any], results: pd.DataFrame) -> list[dict[str, Any]]:
    """After a race: surprises (results the engine gave < SURPRISE_P), upsets (a winner
    under UPSET_WIN_P) and data problems (drivers raced but not predicted, or predicted
    but not starting). results: this race's rows with DriverId, Driver, Finish, Win.

    A surprise is not automatically an error: about 1 driver-result in 100 should land
    there by chance. Many more than that means the model is missing something.
    """
    if results.empty:
        return []
    stem = record.get("_stem", record_stem(record["year"], record["round"], record["moment"]))
    base = {"Stem": stem, "Race": f"{record['year']} {record['event']}",
            "Moment": record["moment"]}
    pred = record_frame(record)
    out: list[dict[str, Any]] = []
    codes = dict(zip(results["DriverId"], results.get("Driver", results["DriverId"])))
    for driver in sorted(set(results["DriverId"]) - set(pred["DriverId"])):
        out.append({**base, "Kind": "data", "Driver": codes.get(driver, driver),
                    "Chance": float("nan"),
                    "Text": "raced but was not in the prediction (new or replacement "
                            "driver: check the entry list)"})
    for driver in sorted(set(pred["DriverId"]) - set(results["DriverId"])):
        name = pred.loc[pred["DriverId"] == driver, "Driver"].iloc[0]
        out.append({**base, "Kind": "data", "Driver": name, "Chance": float("nan"),
                    "Text": "was predicted but did not start (withdrawn, injured or replaced)"})
    merged = pred.merge(results.drop(columns=["Driver"], errors="ignore"), on="DriverId")
    columns = [c for c in pred.columns if c.startswith(POSITION_PREFIX)]
    for row in merged.itertuples(index=False):
        values = row._asdict()
        finish = values.get("Finish")
        if finish is None or pd.isna(finish):
            continue
        k = min(int(finish), len(columns))
        chance = float(sum(values[c] for c in columns[:k]))
        if chance < SURPRISE_P:
            out.append({**base, "Kind": "surprise", "Driver": values["Driver"],
                        "Chance": chance,
                        "Text": f"finished P{int(finish)}; the engine gave {chance:.2%} for "
                                f"P{int(finish)} or better"})
        if values.get("Win") == 1 and values["P_win"] < UPSET_WIN_P:
            out.append({**base, "Kind": "upset", "Driver": values["Driver"],
                        "Chance": float(values["P_win"]),
                        "Text": f"won with only {values['P_win']:.1%} to win"})
    return out


def check_records(records: list[dict[str, Any]], results: pd.DataFrame) -> pd.DataFrame:
    """check_record for every record whose race has results (results needs Year, Round)."""
    rows: list[dict[str, Any]] = []
    for record in records:
        race = results[(results["Year"] == record["year"]) & (results["Round"] == record["round"])]
        rows += check_record(record, race)
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def expected_surprises(graded: pd.DataFrame) -> float:
    """How many surprises chance alone would give: about SURPRISE_P per classified row."""
    if graded.empty or "Finish" not in graded:
        return 0.0
    return float(graded["Finish"].notna().sum() * SURPRISE_P)


def graded_rows(records: list[dict[str, Any]], results: pd.DataFrame) -> pd.DataFrame:
    """Every (record, driver) row with its outcome, for calibration tables."""
    parts = []
    for record in records:
        race = results[(results["Year"] == record["year"]) & (results["Round"] == record["round"])]
        if race.empty:
            continue
        frame = record_frame(record).merge(race.drop(columns=["Driver"], errors="ignore"),
                                           on="DriverId", how="inner")
        frame["Moment"] = record["moment"]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_track_record(scores: pd.DataFrame, graded: pd.DataFrame,
                       path: Path | str = TRACK_RECORD_PATH,
                       pending: list[str] | None = None,
                       checks: pd.DataFrame | None = None) -> Path:
    """reports/track_record.md: every scored prediction, running averages vs targets."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Track record", "",
             "Every live prediction pitwall-f1 made BEFORE the race (records in "
             "`predictions/`, timestamped by their git commit), graded after it.", "",
             "Log losses are averaged over driver rows (lower is better). Targets: the best "
             f"open-source model found reaches {BENCHMARKS['win_logloss']} (win) and "
             f"{BENCHMARKS['podium_logloss']} (podium). B0 = the grid-slot baseline made at "
             "the same moment, from the same data.", ""]
    if scores.empty:
        lines += ["_No race with a prediction has finished yet._", ""]
    else:
        lines += ["## Running averages", "",
                  "| Moment | Races | Win LL | B0 win LL | Podium LL | B0 podium LL | "
                  "Winner hit rate | Podium overlap |",
                  "|---|---|---|---|---|---|---|---|"]
        for moment, group in scores.groupby("Moment", sort=False):
            lines.append(
                f"| {moment} | {len(group)} | {group['win_logloss'].mean():.4f} | "
                f"{_nan(group['b0_win_logloss'].mean())} | "
                f"{group['podium_logloss'].mean():.4f} | "
                f"{_nan(group['b0_podium_logloss'].mean())} | "
                f"{group['WinnerHit'].mean():.0%} | {group['PodiumOverlap'].mean():.0%} |")
        lines += ["", f"Targets: win {BENCHMARKS['win_logloss']}, podium "
                  f"{BENCHMARKS['podium_logloss']}. With few races these averages are very "
                  "noisy: read them as a diary, not a verdict.", "",
                  "## Every prediction", "",
                  "| Race | Moment | Made (UTC) | Win LL | B0 | Podium LL | B0 | Favourite | "
                  "Winner | P(win) of winner |", "|---|---|---|---|---|---|---|---|---|---|"]
        for row in scores.sort_values(["Year", "Round"], ascending=False).itertuples(index=False):
            lines.append(
                f"| {row.Year} R{row.Round} {row.Event} | {row.Moment} | {row.Created[:16]} | "
                f"{row.win_logloss:.4f} | {_nan(row.b0_win_logloss)} | "
                f"{row.podium_logloss:.4f} | {_nan(row.b0_podium_logloss)} | {row.Favourite} | "
                f"{row.Winner}{' ✓' if row.WinnerHit else ''} | {row.P_win_of_winner:.1%} |")
        lines += [""]
        if not graded.empty and len(graded) >= 20:
            table = calibration_table(graded["P_win"], graded["Win"], bins=5)
            lines += ["## Calibration (win, all moments)", "", "When we said about p, the "
                      "favourite won about this often:", "",
                      "| Mean predicted | Observed | Rows |", "|---|---|---|"]
            lines += [f"| {r.Predicted:.3f} | {r.Observed:.3f} | {int(r.Count)} |"
                      for r in table.itertuples(index=False)]
            lines += [""]
    if checks is not None and not scores.empty:
        lines += _check_lines(checks, graded)
    if pending:
        lines += ["## Waiting for results", "", *[f"- {p}" for p in pending], ""]
    lines += ["_Educational project, not betting advice. Not affiliated with Formula 1._", ""]
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _check_lines(checks: pd.DataFrame, graded: pd.DataFrame) -> list[str]:
    """The 'Surprises and data checks' section."""
    surprises = int((checks["Kind"] == "surprise").sum())
    expected = expected_surprises(graded)
    lines = ["## Surprises and data checks", "",
             f"A surprise is a result the engine gave under {SURPRISE_P:.0%} (e.g. a rookie on "
             f"the podium in a slow car). By chance alone about {expected:.1f} are expected "
             f"so far; observed: {surprises}."]
    if surprises > max(3.0, 3 * expected):
        lines.append("**Warning: far more surprises than chance allows. The model is "
                     "probably missing something (a new driver, an upgrade, a rule change); "
                     "investigate before trusting the next predictions.**")
    data = checks[checks["Kind"] == "data"]
    if not data.empty:
        lines.append(f"**{len(data)} data problem(s)** (entry list mismatches) are listed "
                     "below: these are errors in the inputs, not bad luck.")
    lines.append("")
    if checks.empty:
        return lines + ["_No surprises or data problems._", ""]
    lines += ["| Race | Moment | Kind | Driver | What happened |", "|---|---|---|---|---|"]
    for row in checks.itertuples(index=False):
        lines.append(f"| {row.Race} | {row.Moment} | {row.Kind} | {row.Driver} | {row.Text} |")
    return lines + [""]


def _nan(value: float) -> str:
    return "–" if value is None or not np.isfinite(value) else f"{value:.4f}"


# ---------- the GitHub Pages site ----------
def write_docs(records: list[dict[str, Any]], docs: Path | str = DOCS_DIR,
               track_record: Path | str = TRACK_RECORD_PATH) -> Path:
    """docs/index.md listing every prediction (newest first) + one page per prediction."""
    folder = Path(docs)
    (folder / "predictions").mkdir(parents=True, exist_ok=True)
    if Path(track_record).exists():
        (folder / "track-record.md").write_text(Path(track_record).read_text("utf-8"), "utf-8")
    lines = ["# pitwall-f1 race predictions", "",
             "Probabilities for every Formula 1 race, made before the race by an open-source "
             "model and graded afterwards. [Track record](track-record.md) · "
             "[Source code](https://github.com/thanmayroobanathans/formula-1-prediction)", "",
             "_Educational project, not betting advice. Not affiliated with Formula 1._", "",
             "| Race | Moment | Made (UTC) | Favourite | P(win) | Report |",
             "|---|---|---|---|---|---|"]
    for record in reversed(records):
        stem = record.get("_stem", record_stem(record["year"], record["round"],
                                               record["moment"]))
        top = record["drivers"][0] if record["drivers"] else {}
        best = max(record["drivers"], key=lambda d: d["P_win"]) if record["drivers"] else top
        pdf = folder / "reports" / f"{stem}.pdf"
        link = f"[PDF](reports/{stem}.pdf)" if pdf.exists() else ""
        moment = record["moment"] + (f" ({record['tag']})" if record.get("tag") else "")
        lines.append(f"| [{record['year']} {record['event']}](predictions/{stem}.md) | "
                     f"{moment} | {record['created_utc'][:16]} | "
                     f"{best.get('Driver', '?')} | {best.get('P_win', 0):.1%} | {link} |")
        _prediction_page(record, stem, folder)
    if not records:
        lines.append("| _no predictions yet_ | | | | | |")
    lines += ["", "Data: f1db (CC BY 4.0), Jolpica, OpenF1 (CC BY-NC-SA 4.0), Open-Meteo "
              "(CC BY 4.0).", ""]
    index = folder / "index.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    config = folder / "_config.yml"
    if not config.exists():
        config.write_text("title: pitwall-f1\ndescription: Open-source F1 race predictions\n"
                          "theme: jekyll-theme-minimal\n", encoding="utf-8")
    return index


def _prediction_page(record: dict[str, Any], stem: str, folder: Path) -> None:
    tag = f" ({record['tag']})" if record.get("tag") else ""
    lines = [f"# {record['year']} {record['event']} — {record['moment']}{tag}", "",
             f"Made {record['created_utc']} (git {record['git']}, config "
             f"{record['config_hash']}, pitwall-f1 {record['version']}). "
             f"Race date {record['race_date']}.", ""]
    image = folder / "img" / f"{stem}-bars.png"
    if image.exists():
        lines += [f"![Win, podium and points probabilities](../img/{stem}-bars.png)", ""]
    lines += ["| # | Driver | Team | Win | Podium | Points | DNF | Exp. pos |",
              "|---|---|---|---|---|---|---|---|"]
    for i, d in enumerate(record["drivers"], start=1):
        lines.append(f"| {i} | {d['Driver']} | {d.get('TeamId', '')} | {d['P_win']:.1%} | "
                     f"{d['P_podium']:.1%} | {d['P_points']:.1%} | {d['P_dnf']:.1%} | "
                     f"{d['ExpectedPosition']:.1f} |")
    flags = ", ".join(f"{k}={v}" for k, v in record.get("availability", {}).items())
    lines += ["", f"Data availability: {flags}", ""]
    lines += [f"- ⚠ {w}" for w in record.get("warnings", [])]
    lines += [f"- {n}" for n in record.get("notes", [])]
    lines += ["", "[All predictions](../index.md)", ""]
    (folder / "predictions" / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
