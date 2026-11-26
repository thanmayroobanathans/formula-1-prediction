"""Phase 12: the steps behind the CLI (and the GitHub workflows), as plain functions.

Sources are passed in (a `Sources` bundle), so tests run everything with fakes and
never touch the network. Only free sources: f1db, Jolpica, OpenF1 (historical) and
Open-Meteo. The F1 TV token is never used here.
"""
from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ..cache import DEFAULT_CACHE_DIR
from ..errors import DataNotAvailableError
from ..sources.base import ForecastSource, HistorySource, Record, ResultsSource, WeekendSource
from .history import load_history
from .live import LiveRows, ToppedUpHistory, build_live_rows, find_event, next_race
from .pipeline import CONFIG_PATH, FinalPipeline, PipelineConfig, load_config
from .records import (
    DOCS_DIR, RECORDS_DIR, TRACK_RECORD_PATH, RecordMeta, check_records, graded_rows,
    load_records,
    make_record, record_stem, save_record, score_records, write_docs, write_track_record,
)

log = logging.getLogger(__name__)
HISTORY_PREDICTIONS = DEFAULT_CACHE_DIR / "predict" / "backtest_predictions.parquet"


@dataclass
class Sources:
    """The data sources one run uses (fakes in tests)."""

    history: HistorySource
    jolpica: ResultsSource
    weekend: WeekendSource | None = None
    weather: ForecastSource | None = None
    live_forecast: Callable[[float, float, str], list[Record]] | None = None


def default_sources(openf1: bool = True, weather: bool = True) -> Sources:
    """The real, free sources (imported lazily so `--help` stays fast)."""
    from ..sources.f1db import F1dbLoader
    from ..sources.jolpica import JolpicaClient
    from ..sources.openf1 import OpenF1Client
    from ..sources.openmeteo import OpenMeteoClient

    meteo = OpenMeteoClient() if weather else None
    return Sources(F1dbLoader(), JolpicaClient(), OpenF1Client() if openf1 else None, meteo,
                   meteo.get_live_forecast if meteo else None)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


@dataclass
class PredictionOutput:
    """What `predict_race` produced."""

    prediction: pd.DataFrame
    baseline: pd.DataFrame
    live: LiveRows
    record: dict[str, Any]
    path: Path | None
    teams: dict[str, str] = field(default_factory=dict)


def resolve_event(sources: Sources, year: int | str, event: str,
                  now: dt.datetime | None = None) -> tuple[int, str]:
    """('next' or a year, event) -> (year, round as text). 'next' finds the next race."""
    when = now or _utcnow()
    if str(event).lower() != "next" and str(year).lower() != "next":
        return int(year), str(event)
    for season in (when.year, when.year + 1):
        race = next_race(sources.jolpica.get_schedule(season), when.date())
        if race is not None:
            return season, str(race["Round"])
    raise DataNotAvailableError("No upcoming race found in this season or the next.")


def predict_race(year: int, event: str, moment: str, sources: Sources,
                 config: PipelineConfig | None = None,
                 history_predictions: pd.DataFrame | None = None,
                 records_dir: Path | str = RECORDS_DIR, save: bool = True,
                 first_year: int = 2018, now: Callable[[], dt.datetime] | None = None,
                 n_seeds: int | None = None, live_rows: LiveRows | None = None,
                 tag: str = "") -> PredictionOutput:
    """Build the live rows, run the pipeline and B0, and (optionally) save the record.

    The record is checked for existence FIRST, so a rerun costs nothing and never
    overwrites the original prediction.
    """
    clock = now or _utcnow
    schedule = sources.jolpica.get_schedule(year)
    race = find_event(schedule, event)
    stem = record_stem(year, int(race["Round"]), moment, tag)
    target = Path(records_dir) / f"{stem}.json"
    if save and target.exists():
        from .records import RecordExistsError
        raise RecordExistsError(f"{target} already exists (records are never overwritten).")
    config = config or load_config()
    if n_seeds is not None:
        config = PipelineConfig(config.moments, n_seeds, config.note, config.extra)
    if history_predictions is None and HISTORY_PREDICTIONS.exists():
        history_predictions = pd.read_parquet(HISTORY_PREDICTIONS)
    live = live_rows or build_live_rows(year, race["Round"], moment, sources.history,
                                        sources.jolpica, sources.weekend, sources.weather,
                                        sources.live_forecast, first_year=first_year, now=clock)
    pipeline = FinalPipeline(config, history_predictions)
    prediction = pipeline.predict(live.data, live.race_id, moment, live=True)
    baseline_config = PipelineConfig({moment: _single("B0 grid slot")})
    baseline = FinalPipeline(baseline_config).predict(live.data, live.race_id, moment, live=True)
    teams = dict(zip(live.race["DriverId"].astype(str), live.race["TeamId"].astype(str)))
    choice = asdict(config.moments[moment])
    meta = RecordMeta(year, live.round, live.race_id, live.event, str(live.date.date()), moment,
                      {"moment": choice, "n_seeds": config.n_seeds, "note": config.note},
                      live.flags, live.warnings, live.notes)
    record = make_record(prediction, meta, baseline, teams, now=clock())
    record["context"] = race_context(live)
    if tag:
        record["tag"] = tag
        record["notes"] = [*record.get("notes", []),
                           f"Extra prediction '{tag}' with the pipeline above, made after this "
                           "moment's first record; both are kept and graded."]
    path = save_record(record, records_dir) if save else None
    return PredictionOutput(prediction, baseline, live, record, path, teams)


CONTEXT_FEATURES = ["air_temp_forecast", "rain_forecast_mm", "has_forecast", "track_temp",
                    "has_track_temp", "circuit_sc_rate", "has_sc_history", "circuit_street",
                    "circuit_high_speed", "circuit_high_downforce", "circuit_places_moved"]


def race_context(live: LiveRows) -> dict[str, Any]:
    """Race-level facts the model saw (weather forecast, circuit), for records and reports."""
    first = live.race.iloc[0]
    out: dict[str, Any] = {"circuit_id": live.circuit_id}
    for name in CONTEXT_FEATURES:
        value = first.get(name)
        out[name] = None if value is None or pd.isna(value) else round(float(value), 4)
    return out


def _single(name: str) -> Any:
    from .pipeline import MomentChoice
    return MomentChoice([name])


def results_table(sources: Sources, years: list[int],
                  now: Callable[[], dt.datetime] | None = None) -> pd.DataFrame:
    """Labels of every finished race in `years` (f1db topped up from Jolpica)."""
    topped = ToppedUpHistory(sources.history, sources.jolpica, years, now=now)
    hist = load_history(topped, min(years))
    columns = ["Year", "Round", "DriverId", "Driver", "Finish", "Win", "Podium",
               "PointsFinish", "DNF"]
    return hist.results.loc[hist.results["Year"].isin(years), columns]


def score_all(sources: Sources, records_dir: Path | str = RECORDS_DIR,
              track_path: Path | str = TRACK_RECORD_PATH, docs_dir: Path | str | None = DOCS_DIR,
              now: Callable[[], dt.datetime] | None = None) -> pd.DataFrame:
    """Grade every record whose race has results; rewrite the track record and docs."""
    records = load_records(records_dir)
    if not records:
        scores = score_records([], pd.DataFrame(columns=["Year", "Round"]))
        write_track_record(scores, pd.DataFrame(), track_path)
        if docs_dir is not None:
            write_docs(records, docs_dir, track_path)
        return scores
    years = sorted({int(r["year"]) for r in records})
    results = results_table(sources, years, now)
    scores = score_records(records, results)
    graded = graded_rows(records, results)
    done = set(scores["Stem"])
    pending = [f"{r['year']} {r['event']} ({r['moment']})" for r in records
               if r["_stem"] not in done]
    checks = check_records(records, results)
    scores.attrs["checks"] = checks
    write_track_record(scores, graded, track_path, pending, checks)
    if docs_dir is not None:
        _track_plot(scores, Path(docs_dir))
        write_docs(records, docs_dir, track_path)
    return scores


def save_prediction_plots(output: PredictionOutput, docs_dir: Path | str = DOCS_DIR) -> list[Path]:
    """docs/img/<stem>-bars.png and -heatmap.png (light style, for the web page)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from ..plotting.live_plots import position_heatmap, prediction_bars

    folder = Path(docs_dir) / "img"
    folder.mkdir(parents=True, exist_ok=True)
    stem = record_stem(output.live.year, output.live.round, output.live.moment,
                       output.record.get("tag", ""))
    title = f"{output.live.year} {output.live.event} ({output.live.moment})"
    paths = []
    with plt.style.context("default"):
        for suffix, draw in (("bars", lambda: prediction_bars(output.prediction, output.teams,
                                                               title)),
                             ("heatmap", lambda: position_heatmap(output.prediction, title))):
            ax = draw()
            path = folder / f"{stem}-{suffix}.png"
            fig = cast(Figure, ax.figure)
            fig.tight_layout()
            fig.savefig(path, dpi=110)
            plt.close(fig)
            paths.append(path)
    return paths


def _track_plot(scores: pd.DataFrame, docs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from ..plotting.live_plots import track_record_plot

    (docs / "img").mkdir(parents=True, exist_ok=True)
    with plt.style.context("default"):
        ax = track_record_plot(scores)
        fig = cast(Figure, ax.figure)
        fig.tight_layout()
        fig.savefig(docs / "img" / "track-record.png", dpi=110)
        plt.close(fig)


def format_prediction(pred: pd.DataFrame, teams: dict[str, str] | None = None,
                      limit: int | None = None) -> str:
    """A readable table: drivers by expected position, probabilities in %."""
    table = pred.sort_values("ExpectedPosition").reset_index(drop=True)
    shown = pd.DataFrame({
        "#": range(1, len(table) + 1), "Driver": table["Driver"],
        "Team": [(teams or {}).get(str(d), "") for d in table["DriverId"]],
        "Win": table["P_win"].map("{:.1%}".format),
        "Podium": table["P_podium"].map("{:.1%}".format),
        "Points": table["P_points"].map("{:.1%}".format),
        "DNF": table["P_dnf"].map("{:.1%}".format),
        "ExpPos": table["ExpectedPosition"].map("{:.1f}".format)})
    if limit:
        shown = shown.head(limit)
    return shown.to_string(index=False)


__all__ = ["Sources", "default_sources", "PredictionOutput", "resolve_event", "predict_race",
           "results_table", "score_all", "save_prediction_plots", "format_prediction",
           "HISTORY_PREDICTIONS", "CONFIG_PATH"]
