"""Deterministic surface-form normalization for source vocabularies.

Every source spells the same event differently: `Ball Carry`, `carry`,
`CARRY_INTO_CONTACT`, `Pick & Go`. This module folds those surface forms onto
the canonical :data:`ruck.schema.ACTIONS` and :data:`ruck.schema.OUTCOMES`
vocabularies with a single pass of string normalization followed by a table
lookup. No fuzzy matching, no learned mapping: the same input always produces
the same output, and an unrecognised label is reported rather than guessed at.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^a-z0-9]+")


def fold(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace to ``_``."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _PUNCT.sub("_", ascii_only.lower()).strip("_")


#: Surface form (already folded) -> canonical action. Where a source carries a
#: more specific label than the canonical vocabulary, the mapper keeps the
#: original in `subtype`, so nothing is lost by folding.
ACTION_ALIASES: dict[str, str] = {
    # carrying
    "carry": "carry", "ball_carry": "carry", "run": "carry", "runs": "carry",
    "carry_into_contact": "carry", "pick_and_go": "carry", "pick_go": "carry",
    "crash_ball": "carry", "line_break": "carry", "metres_carried": "carry",
    # distribution
    "pass": "pass", "passes": "pass", "ball_pass": "pass", "distribution": "pass",
    "offload": "offload", "offloads": "offload", "offload_in_tackle": "offload",
    # kicking out of hand
    "kick": "kick", "kicks": "kick", "box_kick": "kick", "clearance": "kick",
    "clearance_kick": "kick", "grubber": "kick", "chip": "kick", "chip_kick": "kick",
    "up_and_under": "kick", "garryowen": "kick", "territorial_kick": "kick",
    "cross_kick": "kick", "kick_in_play": "kick", "kick_from_hand": "kick",
    # breakdown and set piece
    "ruck": "ruck", "rucks": "ruck", "breakdown": "ruck", "ruck_won": "ruck",
    "maul": "maul", "mauls": "maul", "driving_maul": "maul", "rolling_maul": "maul",
    "lineout": "lineout", "line_out": "lineout", "lineouts": "lineout", "throw": "lineout",
    "throw_in": "lineout", "lineout_throw": "lineout",
    "scrum": "scrum", "scrums": "scrum", "scrum_feed": "scrum",
    # defence
    "tackle": "tackle", "tackles": "tackle", "tackle_made": "tackle",
    "tackle_completed": "tackle", "hit": "tackle",
    "turnover": "turnover", "turnovers": "turnover", "turnover_won": "turnover",
    "turnover_conceded": "turnover", "jackal": "turnover", "steal": "turnover",
    # discipline and errors
    "penalty": "penalty", "penalties": "penalty", "pen": "penalty",
    "penalty_conceded": "penalty", "penalty_won": "penalty",
    "free_kick": "free_kick", "freekick": "free_kick", "fk": "free_kick",
    "knock_on": "knock_on", "knockon": "knock_on", "knock_ons": "knock_on",
    "handling_error": "knock_on", "fumble": "knock_on",
    "forward_pass": "forward_pass", "fwd_pass": "forward_pass",
    # scoring
    "try": "try", "tries": "try", "try_scored": "try", "penalty_try": "try",
    "conversion": "conversion", "conversions": "conversion", "con": "conversion",
    "conversion_goal": "conversion",
    "penalty_goal": "penalty_goal", "penalty_kick_at_goal": "penalty_goal",
    "pen_goal": "penalty_goal", "shot_at_goal": "penalty_goal",
    "drop_goal": "drop_goal", "dropped_goal": "drop_goal", "dg": "drop_goal",
    "drop_kick_goal": "drop_goal",
    # restarts, stoppages, admin
    "card": "card", "yellow_card": "card", "red_card": "card", "sin_bin": "card",
    "restart": "restart", "kick_off": "restart", "kickoff": "restart",
    "drop_out": "restart", "22_drop_out": "restart", "22m_restart": "restart",
    "touch": "touch", "into_touch": "touch", "kick_to_touch": "touch",
    "ball_out_of_play": "touch",
    "mark": "mark", "mark_called": "mark",
    "substitution": "substitution", "sub": "substitution", "replacement": "substitution",
    "player_substituted": "substitution", "substitute_on": "substitution",
    "substitute_off": "substitution",
    "end_period": "end_period", "half_time": "end_period", "full_time": "end_period",
    "end_of_half": "end_period", "end_of_match": "end_period",
}

#: Surface form (already folded) -> canonical outcome.
OUTCOME_ALIASES: dict[str, str] = {
    "complete": "complete", "completed": "complete", "success": "complete",
    "successful": "complete", "made": "complete", "good": "complete",
    "y": "complete", "yes": "complete", "true": "complete", "1": "complete",
    "incomplete": "incomplete", "unsuccessful": "incomplete", "fail": "incomplete",
    "failed": "incomplete", "missed": "incomplete", "miss": "incomplete",
    "n": "incomplete", "no": "incomplete", "false": "incomplete", "0": "incomplete",
    "retained": "retained", "kept": "retained", "secured": "retained",
    "own_ball": "retained", "ball_retained": "retained", "quick_ball": "retained",
    "lost": "lost", "ball_lost": "lost", "turnover_lost": "lost", "stolen": "lost",
    "won": "won", "win": "won",
    "penalty_won": "penalty_won", "pen_won": "penalty_won",
    "penalty_conceded": "penalty_conceded", "pen_conceded": "penalty_conceded",
    "points": "points", "scored": "points", "score": "points", "try": "points",
    "held": "held", "held_up": "held", "held_up_in_goal": "held",
    "into_touch": "into_touch", "out": "into_touch", "touch": "into_touch",
    "in_field": "in_field", "in_play": "in_field", "regathered": "in_field",
    "unknown": "unknown", "": "unknown", "na": "unknown", "none": "unknown",
}


def map_action(raw: object) -> tuple[str, str | None]:
    """Fold ``raw`` onto a canonical action.

    Returns ``(action, subtype)``. When the folded label is more specific than
    the canonical action, the original label is returned as the subtype;
    unrecognised labels come back as ``("other", <folded label>)`` so the
    caller can count and report them without discarding the row.
    """
    if raw is None:
        return "other", None
    folded = fold(raw)
    if not folded:
        return "other", None
    canonical = ACTION_ALIASES.get(folded)
    if canonical is None:
        return "other", folded
    return canonical, (folded if folded != canonical else None)


def map_outcome(raw: object) -> str:
    """Fold ``raw`` onto a canonical outcome, defaulting to ``unknown``."""
    if raw is None:
        return "unknown"
    return OUTCOME_ALIASES.get(fold(raw), "unknown")


_CLOCK = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")
_MINUTE_MARK = re.compile(r"^(\d+(?:\.\d+)?)\s*'$")


def parse_time(raw: object) -> float | None:
    """Parse a match clock into seconds from kickoff.

    Accepts a bare number of seconds, ``MM:SS``, ``HH:MM:SS``, and the ``46'``
    minute notation that broadcast feeds use. Returns ``None`` for anything
    else rather than guessing.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    minute = _MINUTE_MARK.match(text)
    if minute:
        return float(minute.group(1)) * 60.0
    clock = _CLOCK.match(text)
    if clock:
        hours = float(clock.group(1) or 0)
        return hours * 3600.0 + float(clock.group(2)) * 60.0 + float(clock.group(3))
    try:
        return float(text)
    except ValueError:
        return None
