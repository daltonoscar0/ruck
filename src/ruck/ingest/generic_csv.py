"""Generic analyst-export CSV loader.

This is the ingest path that carries real phase-play data. An analyst exports
from Opta, Stats Perform, SportsCode or a club's own tagging tool, and as long
as the export has a time, a team, an action and ideally a position, ruck reads
it. The canonical column set is documented in ``docs/csv_format.md``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ruck.ingest.vocab import fold, map_action, map_outcome, parse_time
from ruck.schema import PITCH_LENGTH, PITCH_WIDTH, coerce, derive_possessions, partition_valid

#: Canonical name -> the surface header names accepted for it. Matching is done
#: on folded headers, so ``Start X``, ``start_x`` and ``START-X`` all work.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "t": ("t", "time", "timestamp", "secs", "seconds", "clock", "match_time", "game_time"),
    "team": ("team", "team_name", "possession_team", "attacking_team", "side", "squad"),
    "type": ("action", "event", "event_type", "type", "event_name", "descriptor", "code"),
    "x": ("x", "x_m", "x_coord", "pitch_x", "start_x", "x_start"),
    "y": ("y", "y_m", "y_coord", "pitch_y", "start_y", "y_start"),
    "end_x": ("end_x", "x_end", "finish_x", "target_x"),
    "end_y": ("end_y", "y_end", "finish_y", "target_y"),
    "phase": ("phase", "phase_count", "phase_number", "phase_no", "phase_index"),
    "outcome": ("outcome", "result", "event_outcome", "success", "event_result"),
    "period": ("period", "half", "match_period", "period_number"),
    "player": ("player", "player_name", "athlete", "name"),
    "player_id": ("player_id", "athlete_id", "playerid"),
    "team_id": ("team_id", "teamid", "squad_id"),
    "subtype": ("subtype", "detail", "qualifier", "sub_event", "event_detail"),
    "possession_id": ("possession_id", "possession", "sequence", "sequence_id", "set_id"),
    "pressure": ("pressure", "under_pressure", "contested"),
    "match_id": ("match_id", "game_id", "fixture_id", "matchid"),
    "competition": ("competition", "comp", "league", "tournament"),
    "season": ("season", "year", "campaign"),
}

REQUIRED = ("t", "team", "type")


@dataclass
class LoadReport:
    """What a loader did and what it could not do.

    Carried alongside every loaded frame so the CLI can tell an analyst which
    rows were dropped and why, instead of silently shrinking their match.
    """

    source: str
    rows_in: int = 0
    events_out: int = 0
    dropped: list[tuple[int, str]] = field(default_factory=list)
    unmapped_actions: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.source}: {self.events_out}/{self.rows_in} rows kept"]
        if self.dropped:
            parts.append(f"{len(self.dropped)} dropped")
        if self.unmapped_actions:
            top = ", ".join(f"{k} x{v}" for k, v in self.unmapped_actions.most_common(5))
            parts.append(f"unmapped actions: {top}")
        return "; ".join(parts) + ("\n  " + "\n  ".join(self.notes) if self.notes else "")


def resolve_columns(headers: list[str]) -> dict[str, str]:
    """Map canonical names to the actual headers present in ``headers``.

    The first alias that matches wins, so the tables above are ordered
    most-specific first.
    """
    folded = {fold(h): h for h in headers}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in folded:
                resolved[canonical] = folded[alias]
                break
    return resolved


def load_csv(
    path: str | Path,
    *,
    match_id: str | None = None,
    competition: str | None = None,
    season: str | None = None,
    coordinates: str = "auto",
    direction: str = "possession",
    home_team: str | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load a generic analyst CSV into the canonical events frame.

    Args:
        path: the CSV file.
        match_id: overrides / supplies the match identifier when the file has
            no ``match_id`` column. Defaults to the file stem.
        coordinates: ``"metres"`` (x in 0-100, y in 0-70), ``"percent"``
            (both 0-100), or ``"auto"`` to pick by inspecting the data.
        direction: ``"possession"`` if the export is already attack-normalized
            left-to-right (the sportio convention), or ``"absolute"`` if x is
            measured in a fixed camera frame, in which case ``home_team`` is
            required and away-team and second-half coordinates are flipped.
        strict: raise on the first schema violation instead of dropping the row.

    Returns:
        The canonical frame and a :class:`LoadReport`.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    report = LoadReport(source=f"csv:{path.name}", rows_in=len(raw))

    resolved = resolve_columns(list(raw.columns))
    missing = [c for c in REQUIRED if c not in resolved]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}. "
            f"Accepted headers: {', '.join(COLUMN_ALIASES[m][0] for m in missing)}. "
            f"See docs/csv_format.md."
        )

    out = pd.DataFrame(index=raw.index)
    for canonical, header in resolved.items():
        out[canonical] = raw[header]

    out["t"] = out["t"].map(parse_time)
    mapped = out["type"].map(map_action)
    out["type"] = [m[0] for m in mapped]
    fallback_subtype = [m[1] for m in mapped]
    if "subtype" in out.columns:
        out["subtype"] = [
            s if pd.notna(s) and str(s).strip() else f
            for s, f in zip(out["subtype"], fallback_subtype, strict=True)
        ]
    else:
        out["subtype"] = fallback_subtype
    for action, subtype in zip(out["type"], out["subtype"], strict=True):
        if action == "other" and subtype:
            report.unmapped_actions[str(subtype)] += 1

    out["outcome"] = out["outcome"].map(map_outcome) if "outcome" in out.columns else "unknown"

    out = _apply_coordinates(out, coordinates, report)
    out = _apply_direction(out, direction, home_team, report)

    if "pressure" in out.columns:
        out["pressure"] = out["pressure"].map(
            lambda v: None if pd.isna(v) else fold(v) in {"1", "true", "y", "yes", "t"}
        )

    out["match_id"] = _fill(out, "match_id", match_id or path.stem)
    out["competition"] = _fill(out, "competition", competition)
    out["season"] = _fill(out, "season", season)
    if "period" not in out.columns:
        out["period"] = _infer_period(out["t"])
    out["extra"] = json.dumps({"loader": "generic_csv", "file": path.name}, sort_keys=True)

    out = coerce(out).sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)

    if strict:
        from ruck.schema import validate_frame

        events = validate_frame(out)
    else:
        events, bad = partition_valid(out)
        report.dropped = bad
        if bad:
            report.notes.append(
                f"{len(bad)} row(s) failed validation and were dropped; "
                f"first: row {bad[0][0]} - {bad[0][1]}"
            )

    events = derive_possessions(events)
    report.events_out = len(events)
    if events["x"].isna().all():
        report.notes.append("no coordinates in this file; zone-dependent analysis will be coarse")
    return events, report


def _fill(df: pd.DataFrame, column: str, value: str | None) -> pd.Series:
    if column in df.columns:
        existing = df[column]
        return existing.fillna(value) if value is not None else existing
    return pd.Series([value] * len(df), index=df.index, dtype="object")


def _infer_period(t: pd.Series) -> pd.Series:
    """Split at 40 minutes when the export has no period column."""
    return (t.fillna(0.0) >= 40 * 60).astype(int) + 1


def _apply_coordinates(df: pd.DataFrame, mode: str, report: LoadReport) -> pd.DataFrame:
    if mode not in {"auto", "metres", "percent"}:
        raise ValueError(f"coordinates must be auto, metres or percent, not {mode!r}")
    have = [c for c in ("x", "y", "end_x", "end_y") if c in df.columns]
    if not have:
        return df
    for c in have:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if mode == "auto":
        y_values = pd.concat([df[c] for c in have if c.endswith("y")], ignore_index=True) if any(
            c.endswith("y") for c in have
        ) else pd.Series(dtype=float)
        y_max = float(y_values.max()) if len(y_values.dropna()) else 0.0
        mode = "percent" if y_max > PITCH_WIDTH else "metres"
        report.notes.append(f"coordinates read as {mode} (max y = {y_max:.1f})")

    if mode == "percent":
        for c in have:
            df[c] = df[c] * (PITCH_LENGTH if c.endswith("x") else PITCH_WIDTH) / 100.0
    return df


def _apply_direction(
    df: pd.DataFrame, mode: str, home_team: str | None, report: LoadReport
) -> pd.DataFrame:
    if mode == "possession":
        return df
    if mode != "absolute":
        raise ValueError(f"direction must be possession or absolute, not {mode!r}")
    if home_team is None:
        raise ValueError("direction='absolute' needs home_team so away coordinates can be flipped")
    if "x" not in df.columns:
        report.notes.append("direction='absolute' requested but the file has no coordinates")
        return df

    home = fold(home_team)
    period = df["period"] if "period" in df.columns else _infer_period(df["t"])
    is_home = df["team"].map(lambda v: fold(v) == home)
    first_half = period.fillna(1).astype(int) % 2 == 1
    # A team attacks left-to-right when it is at home in an odd period, or away
    # in an even period; otherwise its coordinates are mirrored.
    keep = (is_home & first_half) | (~is_home & ~first_half)
    for xc, yc in (("x", "y"), ("end_x", "end_y")):
        if xc in df.columns:
            df[xc] = df[xc].where(keep, PITCH_LENGTH - df[xc])
        if yc in df.columns:
            df[yc] = df[yc].where(keep, PITCH_WIDTH - df[yc])
    report.notes.append(f"flipped {int((~keep).sum())} row(s) to attack left-to-right")
    return df
