"""Seeded synthetic phase-play generator.

There is no open rugby dataset with phase-level events and pitch coordinates
(see ``TODO.md``), so the demo and the test suite need a corpus that behaves
like one. This module produces matches with the structure phase play actually
has — possessions that start at a set piece, carries that gain a couple of
metres into a ruck, kicks that are chosen by field position, teams with
different width and phase-retention tendencies — so that a language model
trained on it has real regularities to learn and real surprises to find.

It is not a rugby simulator and the numbers it produces are not measurements of
anything. Every match it writes carries ``synthetic=true`` in the event
``extra`` field, the competition name says so, and every artefact downstream
of it repeats the label. Point ruck at your own exports for numbers that mean
something.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ruck.schema import PITCH_LENGTH, PITCH_WIDTH, coerce

SYNTHETIC_COMPETITION = "Synthetic Phase League (generated)"

#: Deliberately fictional clubs. Real team names on generated events would be a
#: screenshot away from looking like a real match report.
TEAMS: tuple[str, ...] = ("Kestrels", "Ravens", "Badgers", "Otters", "Herons", "Stags")


@dataclass(frozen=True)
class Style:
    """How a team plays, as the handful of dials the generator responds to."""

    name: str
    #: Probability a phase is continued with a pass rather than a direct carry.
    pass_rate: float = 0.45
    #: Lateral metres a pass typically moves the ball.
    width: float = 9.0
    #: Probability of kicking rather than continuing to carry, before field
    #: position adjustments.
    kick_rate: float = 0.19
    #: Probability a carry into contact produces clean ball.
    ruck_retention: float = 0.90
    #: Probability of an offload out of a carry.
    offload_rate: float = 0.045
    #: Metres a typical carry gains.
    carry_gain: float = 3.4
    #: Penalty concession rate per phase.
    penalty_rate: float = 0.045


STYLES: dict[str, Style] = {
    "Kestrels": Style("Kestrels", pass_rate=0.58, width=12.0, kick_rate=0.16, offload_rate=0.075),
    "Ravens": Style("Ravens", pass_rate=0.32, width=6.0, kick_rate=0.15, carry_gain=4.1,
                    ruck_retention=0.93),
    "Badgers": Style("Badgers", pass_rate=0.40, width=8.0, kick_rate=0.29, carry_gain=2.9),
    "Otters": Style("Otters", pass_rate=0.50, width=10.5, kick_rate=0.21, offload_rate=0.06),
    "Herons": Style("Herons", pass_rate=0.36, width=7.0, kick_rate=0.25, ruck_retention=0.87,
                    penalty_rate=0.060),
    "Stags": Style("Stags", pass_rate=0.46, width=9.5, kick_rate=0.18, carry_gain=3.6),
}

# Seconds each event type consumes. Rough but consistent, which is all the
# clock needs to be for a timeline.
DURATIONS: dict[str, tuple[float, float]] = {
    "carry": (2.5, 1.0),
    "pass": (1.4, 0.5),
    "offload": (1.2, 0.4),
    "ruck": (3.6, 1.4),
    "maul": (8.0, 3.0),
    "kick": (4.5, 1.5),
    "lineout": (26.0, 8.0),
    "scrum": (34.0, 12.0),
    "restart": (18.0, 5.0),
    "tackle": (1.0, 0.3),
    "turnover": (2.0, 0.8),
    "penalty": (6.0, 2.0),
    "knock_on": (2.0, 0.6),
    "try": (4.0, 1.0),
    "conversion": (45.0, 8.0),
    "penalty_goal": (55.0, 10.0),
    "touch": (5.0, 2.0),
    "end_period": (0.0, 0.0),
}

HALF_SECONDS = 40 * 60


class _Clock:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.t = 0.0

    def advance(self, action: str) -> float:
        mean, spread = DURATIONS.get(action, (2.0, 0.8))
        self.t += max(0.2, float(self.rng.normal(mean, spread)))
        return self.t


def generate_match(
    *,
    seed: int,
    home: str,
    away: str,
    match_id: str | None = None,
    competition: str = SYNTHETIC_COMPETITION,
    season: str = "2024",
) -> pd.DataFrame:
    """Generate one match as a canonical events frame.

    Coordinates follow the suite convention: ``x`` is metres from the
    possessing team's own goal line, so every possession runs left to right.
    """
    rng = np.random.default_rng(seed)
    match_id = match_id or f"syn-{seed:05d}"
    styles = {home: STYLES.get(home, Style(home)), away: STYLES.get(away, Style(away))}
    rows: list[dict[str, object]] = []
    score = {home: 0, away: 0}

    for period in (1, 2):
        clock = _Clock(rng)
        clock.t = 0.0 if period == 1 else float(HALF_SECONDS)
        # The team that did not kick off the first half kicks off the second.
        in_possession = home if period == 1 else away
        start_kind = "restart"
        start_x = 50.0
        possession_id = len([r for r in rows if r["type"] == "restart"])

        while clock.t < period * HALF_SECONDS:
            events, next_state = _possession(
                rng=rng,
                clock=clock,
                team=in_possession,
                opponent=away if in_possession == home else home,
                style=styles[in_possession],
                start_kind=start_kind,
                start_x=start_x,
                period=period,
                possession_id=possession_id,
                match_id=match_id,
                competition=competition,
                season=season,
                score=score,
            )
            rows.extend(events)
            possession_id += 1
            in_possession, start_kind, start_x = next_state
            if clock.t >= period * HALF_SECONDS:
                break

        rows.append(
            _event(
                match_id=match_id,
                competition=competition,
                season=season,
                team=in_possession,
                t=min(clock.t, period * HALF_SECONDS),
                period=period,
                action="end_period",
                outcome="unknown",
                x=None,
                y=None,
                phase=0,
                possession_id=possession_id,
            )
        )

    frame = coerce(pd.DataFrame(rows)).sort_values("t", kind="stable").reset_index(drop=True)
    return frame


def _possession(
    *,
    rng: np.random.Generator,
    clock: _Clock,
    team: str,
    opponent: str,
    style: Style,
    start_kind: str,
    start_x: float,
    period: int,
    possession_id: int,
    match_id: str,
    competition: str,
    season: str,
    score: dict[str, int],
) -> tuple[list[dict[str, object]], tuple[str, str, float]]:
    """Simulate one possession. Returns its events and the next possession's start."""
    events: list[dict[str, object]] = []
    x = float(np.clip(start_x, 2.0, 98.0))
    y = float(rng.uniform(8.0, PITCH_WIDTH - 8.0))
    phase = 0

    def emit(action: str, outcome: str, **kwargs: object) -> None:
        events.append(
            _event(
                match_id=match_id,
                competition=competition,
                season=season,
                team=team,
                t=clock.advance(action),
                period=period,
                action=action,
                outcome=outcome,
                x=round(x, 2),
                y=round(y, 2),
                phase=phase,
                possession_id=possession_id,
                **kwargs,
            )
        )

    emit(start_kind, "won" if start_kind in {"lineout", "scrum"} else "complete")

    max_phases = 40
    while phase < max_phases and clock.t < period * HALF_SECONDS:
        # Field position drives the choice more than anything else: teams kick
        # from their own 22 and carry inside the opposition 22.
        kick_pressure = 1.9 if x < 25 else (0.25 if x > 75 else 1.0)
        p_kick = float(np.clip(style.kick_rate * kick_pressure, 0.0, 0.7))

        if rng.random() < p_kick:
            gain = float(rng.normal(34.0, 9.0))
            x_after = float(np.clip(x + gain, 0.0, PITCH_LENGTH))
            to_touch = rng.random() < 0.45
            emit("kick", "into_touch" if to_touch else "in_field", end_x=round(x_after, 2))
            if to_touch:
                # Lineout to the other side, roughly where the ball went out.
                return events, (opponent, "lineout", PITCH_LENGTH - x_after)
            contest = rng.random()
            if contest < 0.30:
                return events, (team, "lineout", x_after)
            return events, (opponent, "restart", PITCH_LENGTH - x_after)

        # A phase is a short chain of passes and then a carry. Wide teams string
        # more passes together; nobody strings more than three.
        for _ in range(min(3, int(rng.poisson(style.pass_rate * 2.2)))):
            direction = 1.0 if rng.random() < 0.5 else -1.0
            shift = direction * abs(rng.normal(style.width, 3.0))
            y = float(np.clip(y + shift, 1.0, PITCH_WIDTH - 1.0))
            if rng.random() < 0.012:
                emit("forward_pass", "lost")
                return events, (opponent, "scrum", PITCH_LENGTH - x)
            emit("pass", "complete")

        # Defences compress as the attack nears the line, so the same carry
        # gains much less inside the 22 than it does at halfway.
        compression = float(np.clip(1.0 - max(0.0, x - 72.0) / 46.0, 0.30, 1.0))
        gain = float(rng.normal(style.carry_gain * compression, 2.6 * compression))
        x = float(np.clip(x + gain, 0.5, PITCH_LENGTH))
        y = float(np.clip(y + rng.normal(0.0, 2.0), 1.0, PITCH_WIDTH - 1.0))

        if x >= PITCH_LENGTH - 0.01 and rng.random() < 0.28:
            emit("carry", "points")
            score[team] += 5
            emit("try", "points")
            converted = rng.random() < 0.72
            emit("conversion", "points" if converted else "incomplete")
            if converted:
                score[team] += 2
            return events, (opponent, "restart", 50.0)

        emit("carry", "retained" if rng.random() < style.ruck_retention else "held")

        if rng.random() < style.offload_rate:
            y = float(np.clip(y + rng.normal(0.0, 4.0), 1.0, PITCH_WIDTH - 1.0))
            emit("offload", "complete")
            continue

        emit("tackle", "complete", by_opponent=True)

        roll = rng.random()
        if roll < 0.032:
            emit("knock_on", "lost")
            return events, (opponent, "scrum", PITCH_LENGTH - x)
        if roll < 0.032 + style.penalty_rate:
            # Penalty to the attacking side: kick at goal from close range,
            # otherwise to the corner.
            emit("penalty", "penalty_won")
            if x > 62 and rng.random() < 0.55:
                made = rng.random() < 0.78
                emit("penalty_goal", "points" if made else "incomplete")
                if made:
                    score[team] += 3
                return events, (opponent, "restart", 50.0)
            return events, (team, "lineout", min(x + 28.0, 96.0))
        if roll < 0.032 + style.penalty_rate + 0.030:
            emit("turnover", "lost")
            return events, (opponent, "ruck", PITCH_LENGTH - x)

        phase += 1
        if rng.random() < 0.07:
            emit("maul", "retained")
            x = float(np.clip(x + abs(rng.normal(2.5, 1.5)), 0.5, PITCH_LENGTH))
        else:
            emit("ruck", "retained")

    return events, (opponent, "scrum", PITCH_LENGTH - x)


def _event(
    *,
    match_id: str,
    competition: str,
    season: str,
    team: str,
    t: float,
    period: int,
    action: str,
    outcome: str,
    x: float | None,
    y: float | None,
    phase: int,
    possession_id: int,
    end_x: float | None = None,
    by_opponent: bool = False,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "competition": competition,
        "season": season,
        "team_id": None,
        "team": team,
        "player_id": None,
        "player": None,
        "t": round(float(t), 2),
        "period": period,
        "type": action,
        "subtype": None,
        "x": x,
        "y": y,
        "end_x": end_x,
        "end_y": None,
        "outcome": outcome,
        "pressure": None,
        "body_part": None,
        "extra": json.dumps(
            {"synthetic": True, "generator": "ruck.demo_data.generate", "by_opponent": by_opponent},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "phase": phase,
        "possession_id": possession_id,
    }


def generate_corpus(
    *, n_matches: int = 100, seed: int = 20240203, competition: str = SYNTHETIC_COMPETITION
) -> pd.DataFrame:
    """Generate a corpus of matches, every pairing drawn from :data:`TEAMS`."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_matches):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        frames.append(
            generate_match(
                seed=int(seed + i),
                home=str(home),
                away=str(away),
                match_id=f"syn-{seed}-{i:03d}",
                competition=competition,
            )
        )
    return coerce(pd.concat(frames, ignore_index=True))


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a match in the documented generic CSV format (``docs/csv_format.md``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[
        ["t", "team", "type", "x", "y", "phase", "outcome", "period", "possession_id", "match_id",
         "competition", "season"]
    ].rename(columns={"type": "action"})
    out.to_csv(path, index=False)
    return path


def default_styles() -> dict[str, Style]:
    """A copy of the style table, for callers that want to vary it."""
    return {name: replace(style) for name, style in STYLES.items()}
