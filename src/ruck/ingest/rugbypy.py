"""RugbyPy-format loader.

`rugbypy` publishes match, team and player *summary* frames — 22, 59 and 54
columns of match totals respectively. There are no events in it, no
coordinates and no phase counts, so it cannot produce a phase-play corpus on
its own.

What it is genuinely good for is context: competition, season, venue, referee,
final score, and the per-team totals that the attack-shape section compares a
match against. This module reads a RugbyPy-format file for exactly that, and
maps event-shaped rows when a file happens to carry them (a club that keeps its
own tagging in RugbyPy's column style, for instance) rather than assuming it
never can.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ruck.ingest.generic_csv import LoadReport, load_csv, resolve_columns


class NoEventDataError(RuntimeError):
    """Raised when a RugbyPy-format file has summary rows but no events."""


#: Columns that identify a rugbypy `fetch_match_details` frame.
MATCH_DETAIL_COLUMNS = frozenset({"match_id", "home_team", "away_team", "home_score", "away_score"})

#: Columns that identify a rugbypy `fetch_team_stats` frame.
TEAM_STAT_COLUMNS = frozenset({"team", "match_id", "tackles", "rucks_won", "metres_carried"})


@dataclass
class MatchContext:
    """Match-level facts pulled from a RugbyPy summary frame."""

    match_id: str
    competition: str | None = None
    season: str | None = None
    date: str | None = None
    venue: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    home_score: int | None = None
    away_score: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def read_frame(path: str | Path) -> pd.DataFrame:
    """Read a RugbyPy-format file. Parquet and CSV are both supported."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def looks_like_events(df: pd.DataFrame) -> bool:
    """True when the frame has the time/team/action triple ruck needs."""
    resolved = resolve_columns(list(df.columns))
    return all(c in resolved for c in ("t", "team", "type"))


def load_rugbypy(
    path: str | Path, *, match_id: str | None = None, strict: bool = False
) -> tuple[pd.DataFrame, LoadReport]:
    """Load a RugbyPy-format file as events.

    Raises:
        NoEventDataError: when the file is a summary frame, which is what
            `rugbypy` itself produces. The message names the alternatives
            rather than returning an empty frame that looks like a bad match.
    """
    path = Path(path)
    frame = read_frame(path)
    if looks_like_events(frame):
        if path.suffix.lower() in {".parquet", ".pq"}:
            # Reuse the CSV mapper by round-tripping through a temporary CSV-like
            # frame; the mapper works on column names, not on file format.
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
                frame.to_csv(handle.name, index=False)
                events, report = load_csv(
                    handle.name, match_id=match_id or path.stem, strict=strict
                )
            report.source = f"rugbypy:{path.name}"
            return events, report
        events, report = load_csv(path, match_id=match_id or path.stem, strict=strict)
        report.source = f"rugbypy:{path.name}"
        return events, report

    kind = describe(frame)
    raise NoEventDataError(
        f"{path.name} is a RugbyPy {kind} frame: summary totals, no events. "
        "rugbypy does not publish event-level data with coordinates. Use it via "
        "match_context() for competition and score context, and load phase play "
        "from an analyst CSV export (docs/csv_format.md) or ESPN."
    )


def describe(df: pd.DataFrame) -> str:
    """Name the kind of RugbyPy frame this is, for error messages."""
    columns = set(df.columns)
    if columns >= TEAM_STAT_COLUMNS:
        return "team-stats"
    if columns >= MATCH_DETAIL_COLUMNS:
        return "match-details"
    if {"player_id", "name", "carries"} <= columns:
        return "player-stats"
    return "summary"


def match_context(
    path_or_frame: str | Path | pd.DataFrame, match_id: str | None = None
) -> MatchContext:
    """Extract match-level context from a RugbyPy match-details frame."""
    frame = path_or_frame if isinstance(path_or_frame, pd.DataFrame) else read_frame(path_or_frame)
    if frame.empty:
        raise ValueError("empty RugbyPy frame")
    if match_id is None:
        row = frame.iloc[0]
    else:
        row = frame.loc[frame["match_id"].astype(str) == str(match_id)].iloc[0]

    def get(column: str) -> object | None:
        if column not in frame.columns:
            return None
        value = row[column]
        return None if pd.isna(value) else value

    def as_int(column: str) -> int | None:
        value = get(column)
        return None if value is None else int(value)

    return MatchContext(
        match_id=str(get("match_id") or match_id or "rugbypy-match"),
        competition=_as_str(get("competition")),
        season=_as_str(get("season")),
        date=_as_str(get("date") or get("game_date")),
        venue=_as_str(get("venue")),
        home_team=_as_str(get("home_team")),
        away_team=_as_str(get("away_team")),
        home_score=as_int("home_score"),
        away_score=as_int("away_score"),
    )


def _as_str(value: object | None) -> str | None:
    return None if value is None else str(value)
