"""Canonical rugby event schema.

The suite's shared data layer is `sportio`; its `events` frame is the contract
every tool in the suite reads. `sportio` is not published yet, so this module
carries the canonical column list verbatim and adds the two rugby-specific
columns phase play needs (`phase`, `possession_id`).

If `sportio` ever becomes importable, :func:`load_sportio_events` hands off to
it and the rest of ruck is unchanged.

Pitch convention (rugby, per the sportio spec): ``x`` is metres from the
possessing team's own goal line in ``[0, 100]``, ``y`` is metres across the
pitch in ``[0, 70]``. Every event is stored attacking left-to-right for the
team in possession, so ``x = 95`` always means "near the opposition line"
regardless of which end the team is playing into.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

PITCH_LENGTH = 100.0
PITCH_WIDTH = 70.0

#: The canonical `sportio.events` columns, in order. Do not reorder: downstream
#: suite tools index positionally in a couple of places.
SPORTIO_COLUMNS: tuple[str, ...] = (
    "match_id",
    "competition",
    "season",
    "team_id",
    "team",
    "player_id",
    "player",
    "t",
    "period",
    "type",
    "subtype",
    "x",
    "y",
    "end_x",
    "end_y",
    "outcome",
    "pressure",
    "body_part",
    "extra",
)

#: Rugby-only additions. `phase` is the phase count inside the current
#: possession (0 = the set piece or restart that started it); `possession_id`
#: groups events between turnovers/stoppages.
RUGBY_COLUMNS: tuple[str, ...] = ("phase", "possession_id")

EVENT_COLUMNS: tuple[str, ...] = SPORTIO_COLUMNS + RUGBY_COLUMNS

#: Event `type` vocabulary. Anything a mapper cannot place lands in `other` with
#: the source's own label preserved in `subtype`, so nothing is silently dropped
#: and nothing is invented.
ACTIONS: tuple[str, ...] = (
    "carry",
    "pass",
    "offload",
    "kick",
    "ruck",
    "maul",
    "lineout",
    "scrum",
    "tackle",
    "turnover",
    "penalty",
    "free_kick",
    "knock_on",
    "forward_pass",
    "try",
    "conversion",
    "penalty_goal",
    "drop_goal",
    "card",
    "restart",
    "touch",
    "mark",
    "substitution",
    "end_period",
    "other",
)

#: Event `outcome` vocabulary.
OUTCOMES: tuple[str, ...] = (
    "complete",
    "incomplete",
    "retained",
    "lost",
    "won",
    "penalty_won",
    "penalty_conceded",
    "points",
    "held",
    "into_touch",
    "in_field",
    "unknown",
)

#: Actions that put points on the board, and how many.
POINTS: dict[str, int] = {
    "try": 5,
    "conversion": 2,
    "penalty_goal": 3,
    "drop_goal": 3,
}

#: Actions that end a possession when they occur.
#:
#: Scoring actions are deliberately absent. A try is followed by a conversion
#: from the same side, and the restart after it opens the next possession on
#: its own — ending the possession at the try would leave the conversion as a
#: one-event possession and drag every per-possession average down.
POSSESSION_ENDING: frozenset[str] = frozenset(
    {"turnover", "knock_on", "forward_pass", "end_period"}
)

_DTYPES: dict[str, str] = {
    "match_id": "string",
    "competition": "string",
    "season": "string",
    "team_id": "string",
    "team": "string",
    "player_id": "string",
    "player": "string",
    "t": "float64",
    "period": "Int64",
    "type": "string",
    "subtype": "string",
    "x": "float64",
    "y": "float64",
    "end_x": "float64",
    "end_y": "float64",
    "outcome": "string",
    "pressure": "boolean",
    "body_part": "string",
    "extra": "string",
    "phase": "Int64",
    "possession_id": "Int64",
}


class RugbyEvent(BaseModel):
    """One normalized rugby event.

    Coordinates are optional on purpose: several real sources (ESPN's rugby
    play feed among them) publish timed events with no spatial information at
    all. A mapper must be able to emit a truthful coordinate-free event rather
    than guess a location.
    """

    model_config = ConfigDict(extra="forbid")

    match_id: str
    competition: str | None = None
    season: str | None = None
    team_id: str | None = None
    team: str | None = None
    player_id: str | None = None
    player: str | None = None
    t: float = Field(ge=0.0, description="seconds from kickoff of the match")
    period: int | None = Field(default=None, ge=1, le=5)
    type: str
    subtype: str | None = None
    x: float | None = Field(default=None, ge=0.0, le=PITCH_LENGTH)
    y: float | None = Field(default=None, ge=0.0, le=PITCH_WIDTH)
    end_x: float | None = Field(default=None, ge=0.0, le=PITCH_LENGTH)
    end_y: float | None = Field(default=None, ge=0.0, le=PITCH_WIDTH)
    outcome: str | None = None
    pressure: bool | None = None
    body_part: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    phase: int | None = Field(default=None, ge=0)
    possession_id: int | None = Field(default=None, ge=0)

    @field_validator("type")
    @classmethod
    def _known_action(cls, v: str) -> str:
        if v not in ACTIONS:
            raise ValueError(f"unknown action {v!r}; use 'other' and keep the label in subtype")
        return v

    @field_validator("outcome")
    @classmethod
    def _known_outcome(cls, v: str | None) -> str | None:
        if v is not None and v not in OUTCOMES:
            raise ValueError(f"unknown outcome {v!r}; allowed: {', '.join(OUTCOMES)}")
        return v


def empty_events() -> pd.DataFrame:
    """An empty frame with the canonical columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=_DTYPES[c]) for c in EVENT_COLUMNS})


def events_to_frame(events: list[RugbyEvent]) -> pd.DataFrame:
    """Build a canonical frame from validated events, sorted by time."""
    if not events:
        return empty_events()
    rows = []
    for e in events:
        row = e.model_dump()
        row["extra"] = json.dumps(row["extra"], sort_keys=True, separators=(",", ":"))
        rows.append(row)
    df = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    df = df.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
    return coerce(df)


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Force the canonical columns and dtypes onto ``df``.

    Missing canonical columns are added as all-null. Extra columns are dropped,
    because downstream suite tools assume the contract exactly.
    """
    out = df.copy()
    for c in EVENT_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[list(EVENT_COLUMNS)]
    for c, dt in _DTYPES.items():
        if dt == "float64":
            # to_numeric turns pd.NA into NaN, which plain float64 can hold;
            # a direct astype cannot.
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        elif dt == "Int64":
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
        elif dt == "boolean":
            out[c] = out[c].map(lambda v: pd.NA if pd.isna(v) else bool(v)).astype("boolean")
        else:
            out[c] = out[c].astype(dt)
    return out


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate every row through :class:`RugbyEvent`; raise on the first failure.

    Used by the loaders' strict path and by the test suite. Mappers running in
    fail-soft mode call :func:`partition_valid` instead.
    """
    valid, bad = partition_valid(df)
    if bad:
        idx, err = bad[0]
        raise ValueError(f"row {idx} failed schema validation: {err}")
    return valid


def partition_valid(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Split ``df`` into rows that validate and a list of ``(index, error)``.

    Fail-soft ingestion keeps the good rows and reports the rest; it never
    repairs a row by inventing a value.
    """
    if df.empty:
        return coerce(df), []
    keep: list[int] = []
    bad: list[tuple[int, str]] = []
    for idx, row in df.iterrows():
        payload = {k: (None if pd.isna(v) else v) for k, v in row.items() if k in _DTYPES}
        extra = payload.get("extra")
        if isinstance(extra, str):
            try:
                payload["extra"] = json.loads(extra)
            except json.JSONDecodeError:
                payload["extra"] = {"raw": extra}
        elif extra is None:
            payload["extra"] = {}
        for numeric in ("t", "x", "y", "end_x", "end_y"):
            if payload.get(numeric) is not None:
                payload[numeric] = float(payload[numeric])
        for integral in ("period", "phase", "possession_id"):
            if payload.get(integral) is not None:
                payload[integral] = int(payload[integral])
        if payload.get("pressure") is not None:
            payload["pressure"] = bool(payload["pressure"])
        try:
            RugbyEvent(**payload)
        except Exception as exc:  # pydantic ValidationError or a cast failure
            bad.append((int(idx), str(exc).replace("\n", " ")[:200]))
        else:
            keep.append(int(idx))
    return coerce(df.loc[keep]).reset_index(drop=True), bad


def derive_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """Fill `possession_id` and `phase` where a source did not supply them.

    A possession starts at kickoff, at any set piece or restart, and after any
    possession-ending event. Phase 0 is the set piece; each subsequent ruck or
    maul increments the phase count, which is how analysts count phases.

    Rows that already carry both columns are left alone, so a source that knows
    its own phase counts always wins over this reconstruction.
    """
    if df.empty:
        return df
    out = df.copy()
    known = out["possession_id"].notna() & out["phase"].notna()
    if bool(known.all()):
        return out

    possession_ids: list[int] = []
    phases: list[int] = []
    pid = 0
    phase = 0
    prev_team: str | None = None
    prev_period: object = None
    started = False
    # True when the previous event closed a possession, so whatever comes next
    # opens a new one at phase 0 whatever kind of event it is. Without this a
    # ruck straight after a turnover is numbered phase 1, and every phase in
    # that possession is off by one.
    fresh = True
    for _, row in out.iterrows():
        action = row["type"]
        team = None if pd.isna(row["team"]) else str(row["team"])
        period = None if pd.isna(row["period"]) else int(row["period"])
        new_period = period is not None and prev_period is not None and period != prev_period

        starts_possession = action in {"lineout", "scrum", "restart"} or not started
        if starts_possession or new_period:
            if started:
                pid += 1
            phase = 0
            started = True
        elif fresh:
            phase = 0
        elif team is not None and prev_team is not None and team != prev_team:
            # Possession changed hands without a set piece: a turnover in play.
            pid += 1
            phase = 0
        elif action in {"ruck", "maul"}:
            phase += 1
        fresh = False

        possession_ids.append(pid)
        phases.append(phase)
        if action in POSSESSION_ENDING:
            pid += 1
            phase = 0
            prev_team = None
            fresh = True
        else:
            prev_team = team if team is not None else prev_team
        if period is not None:
            prev_period = period

    out["possession_id"] = out["possession_id"].fillna(pd.Series(possession_ids, index=out.index))
    out["phase"] = out["phase"].fillna(pd.Series(phases, index=out.index))
    return coerce(out)


def load_sportio_events(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Delegate to `sportio.load_events` if the shared package is installed.

    Raises :class:`ModuleNotFoundError` with a useful message otherwise; every
    ruck loader works standalone, so this is an opt-in fast path only.
    """
    try:
        import sportio  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the env
        raise ModuleNotFoundError(
            "sportio is not installed; ruck's own loaders in ruck.ingest cover "
            "generic CSV, RugbyPy and ESPN without it"
        ) from exc
    return coerce(sportio.load_events(*args, **kwargs))
