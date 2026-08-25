"""ESPN rugby loader.

ESPN's rugby endpoints are unofficial and undocumented, so this loader caches
aggressively and fails soft. It is also honest about what the feed contains:
the play collection carries scoring, card and personnel events with a match
clock and nothing else. There are no pitch coordinates and no phase counts in
it, so events land with ``x``/``y`` null and the resulting frame supports a
scoring timeline rather than phase-play analysis. See ``TODO.md``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from ruck.ingest.generic_csv import LoadReport
from ruck.ingest.http import Fetcher
from ruck.ingest.vocab import map_action
from ruck.schema import coerce, derive_possessions, partition_valid

SITE_API = "https://site.api.espn.com/apis/site/v2/sports/rugby"
CORE_API = "https://sports.core.api.espn.com/v2/sports/rugby/leagues"

#: League slug -> ESPN league id, confirmed against
#: ``sports.core.api.espn.com/v2/sports/rugby/leagues``. Any other ESPN rugby
#: league id may be passed straight through as a string.
LEAGUES: dict[str, str] = {
    "six_nations": "180659",
    "rugby_world_cup": "164205",
    "rugby_championship": "244293",
    "champions_cup": "271937",
    "challenge_cup": "272073",
    "premiership": "267979",
    "urc": "270557",
    "top14": "270559",
    "super_rugby_pacific": "242041",
    "currie_cup": "270555",
    "lions_tour": "268565",
}

#: ESPN play type -> (canonical action, canonical outcome).
PLAY_TYPES: dict[str, tuple[str, str]] = {
    "try": ("try", "points"),
    "penalty try": ("try", "points"),
    "conversion": ("conversion", "points"),
    "conversion missed": ("conversion", "incomplete"),
    "penalty goal": ("penalty_goal", "points"),
    "penalty missed": ("penalty_goal", "incomplete"),
    "drop goal": ("drop_goal", "points"),
    "yellow card": ("card", "penalty_conceded"),
    "red card": ("card", "penalty_conceded"),
    "player substituted": ("substitution", "unknown"),
    "substitute on": ("substitution", "unknown"),
    "substitute off": ("substitution", "unknown"),
}

_ID_IN_REF = re.compile(r"/(?:teams|athletes)/(\d+)")


def resolve_league(league: str) -> str:
    """Accept a slug from :data:`LEAGUES` or a raw ESPN league id."""
    if league in LEAGUES:
        return LEAGUES[league]
    if league.isdigit():
        return league
    raise ValueError(f"unknown league {league!r}; try one of {', '.join(sorted(LEAGUES))} or an id")


def list_matches(league: str, dates: str, fetcher: Fetcher | None = None) -> list[dict[str, str]]:
    """List matches on a date or date range (``YYYYMMDD`` or ``YYYYMMDD-YYYYMMDD``)."""
    fetcher = fetcher or Fetcher()
    payload = fetcher.get_json(f"{SITE_API}/{resolve_league(league)}/scoreboard", {"dates": dates})
    return [
        {"event_id": e["id"], "date": e.get("date", ""), "name": e.get("name", "")}
        for e in payload.get("events", [])
    ]


def load_espn(
    event_id: str,
    league: str = "six_nations",
    fetcher: Fetcher | None = None,
    *,
    strict: bool = False,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load one ESPN match into the canonical events frame.

    Costs two requests (summary, then plays), both cached.
    """
    fetcher = fetcher or Fetcher()
    league_id = resolve_league(league)
    report = LoadReport(source=f"espn:{league}:{event_id}")

    summary = fetcher.get_json(f"{SITE_API}/{league_id}/summary", {"event": event_id})
    context = _match_context(summary, event_id)
    plays = _fetch_plays(league_id, event_id, fetcher)
    report.rows_in = len(plays)
    if not plays:
        report.notes.append("ESPN returned no plays for this event")
        return coerce(pd.DataFrame()), report

    rows: list[dict[str, Any]] = []
    for play in plays:
        rows.append(_map_play(play, context, report))

    frame = coerce(pd.DataFrame(rows))
    frame = frame.sort_values(["period", "t"], kind="stable").reset_index(drop=True)

    if strict:
        from ruck.schema import validate_frame

        events = validate_frame(frame)
    else:
        events, bad = partition_valid(frame)
        report.dropped = bad
        if bad:
            report.notes.append(f"{len(bad)} play(s) failed validation and were dropped")

    events = derive_possessions(events)
    report.events_out = len(events)
    report.notes.append(
        "ESPN's rugby play feed carries scoring, card and substitution events only: "
        "no coordinates, no phase counts"
    )
    return events, report


# -- internals -----------------------------------------------------------


def _fetch_plays(league_id: str, event_id: str, fetcher: Fetcher) -> list[dict[str, Any]]:
    url = f"{CORE_API}/{league_id}/events/{event_id}/competitions/{event_id}/plays"
    collected: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = fetcher.get_json(url, {"limit": 1000, "page": page})
        items = payload.get("items", [])
        for item in items:
            # The core API sometimes returns bare references rather than
            # inlined objects; resolve those one at a time, still rate limited.
            if set(item) <= {"$ref"}:
                item = fetcher.get_json(item["$ref"])
            collected.append(item)
        if page >= int(payload.get("pageCount", 1) or 1):
            break
        page += 1
    return collected


def _match_context(summary: dict[str, Any], event_id: str) -> dict[str, Any]:
    header = summary.get("header", {})
    competitions = header.get("competitions") or [{}]
    competition = competitions[0]

    teams: dict[str, dict[str, str]] = {}
    for competitor in competition.get("competitors", []):
        team = competitor.get("team", {})
        tid = str(team.get("id", competitor.get("id", "")))
        teams[tid] = {
            "team": team.get("displayName") or team.get("name") or tid,
            "home_away": competitor.get("homeAway", ""),
        }

    athletes: dict[str, str] = {}
    for roster in summary.get("rosters", []):
        for entry in roster.get("roster", []):
            athlete = entry.get("athlete", {})
            aid = str(athlete.get("id", ""))
            if aid:
                athletes[aid] = athlete.get("displayName") or athlete.get("fullName") or aid

    season = header.get("season", {})
    return {
        "match_id": f"espn-{event_id}",
        "competition": (header.get("league") or {}).get("name"),
        "season": str(season.get("year")) if season.get("year") else None,
        "date": competition.get("date"),
        "teams": teams,
        "athletes": athletes,
    }


def _map_play(play: dict[str, Any], context: dict[str, Any], report: LoadReport) -> dict[str, Any]:
    label = str((play.get("type") or {}).get("text", "")).strip()
    known = PLAY_TYPES.get(label.lower())
    if known is None:
        action, subtype = map_action(label)
        outcome = "unknown"
        if action == "other":
            report.unmapped_actions[subtype or label] += 1
    else:
        action, outcome = known
        folded = label.lower().replace(" ", "_")
        subtype = None if folded == action else folded

    period = ((play.get("period") or {}).get("number")) or 1
    clock = (play.get("clock") or {}).get("value")
    t = float(clock) if clock is not None else float(period) * 40.0 * 60.0

    team_id = _ref_id(play.get("team"))
    team = context["teams"].get(team_id, {}).get("team")

    player_id = None
    player = None
    for participant in play.get("participants") or []:
        if participant.get("type") in (None, "player"):
            player_id = _ref_id(participant.get("athlete"))
            player = context["athletes"].get(player_id or "")
            break

    return {
        "match_id": context["match_id"],
        "competition": context["competition"],
        "season": context["season"],
        "team_id": team_id,
        "team": team,
        "player_id": player_id,
        "player": player,
        "t": t,
        "period": period,
        "type": action,
        "subtype": subtype,
        "x": None,
        "y": None,
        "end_x": None,
        "end_y": None,
        "outcome": outcome,
        "pressure": None,
        "body_part": None,
        "extra": json.dumps(
            {
                "loader": "espn",
                "espn_play_id": play.get("id"),
                "espn_type": label,
                "home_score": play.get("homeScore"),
                "away_score": play.get("awayScore"),
                "date": context.get("date"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "phase": None,
        "possession_id": None,
    }


def _ref_id(node: Any) -> str | None:
    if not isinstance(node, dict):
        return None
    if node.get("id"):
        return str(node["id"])
    ref = node.get("$ref")
    if not ref:
        return None
    match = _ID_IN_REF.search(str(ref))
    return match.group(1) if match else None
