"""Per-event surprisal, the phase timeline, and the unexpected-plays list.

Surprisal is ``-log2 P(event | everything before it)`` under the phase-play
language model. High surprisal means the model, having read this match and this
possession so far, did not expect what happened next — which is exactly the
clip an analyst wants to watch.

Descriptions are template-generated from the token fields. The model never
writes prose: it produces a number, and the number selects a row from a fixed
table of phrases. There is nothing in a description that is not in the event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ruck.lm.model import PhaseLM
from ruck.tokenizer import Tokenizer, token_of, zone_of

#: Seconds of lead-in to put in front of a clip start, so the cut shows the
#: build-up rather than opening on the moment itself.
CLIP_LEAD_IN = 8.0

ZONE_PHRASE: dict[str, str] = {
    "own22": "inside their own 22",
    "own40": "in their own half",
    "mid": "around halfway",
    "opp40": "in the opposition half",
    "opp22": "inside the opposition 22",
    "opp5": "on the opposition line",
}
CHANNEL_PHRASE: dict[str, str] = {"L": "left channel", "M": "midfield", "R": "right channel"}

ACTION_PHRASE: dict[str, str] = {
    "carry": "carry",
    "pass": "pass",
    "offload": "offload",
    "kick": "kick",
    "ruck": "ruck",
    "maul": "maul",
    "lineout": "lineout",
    "scrum": "scrum",
    "tackle": "tackle",
    "turnover": "turnover",
    "penalty": "penalty",
    "free_kick": "free kick",
    "knock_on": "knock-on",
    "forward_pass": "forward pass",
    "try": "try",
    "conversion": "conversion",
    "penalty_goal": "penalty goal",
    "drop_goal": "drop goal",
    "card": "card",
    "restart": "restart",
    "touch": "kick to touch",
    "mark": "mark",
    "substitution": "substitution",
    "end_period": "end of period",
    "other": "event",
}

OUTCOME_PHRASE: dict[str, str] = {
    "complete": "completed",
    "incomplete": "not completed",
    "retained": "ball retained",
    "lost": "ball lost",
    "won": "won",
    "penalty_won": "penalty won",
    "penalty_conceded": "penalty conceded",
    "points": "points scored",
    "held": "held up",
    "into_touch": "into touch",
    "in_field": "kept in field",
    "unknown": "outcome not recorded",
}


@dataclass
class Highlight:
    """One unexpected passage of play, ready to be cut from video."""

    rank: int
    t: float
    clip_start: float
    period: int
    team: str
    phase: int
    total_bits: float
    peak_bits: float
    n_events: int
    description: str
    event_index: int

    @property
    def clock(self) -> str:
        return format_clock(self.t)

    @property
    def clip_clock(self) -> str:
        return format_clock(self.clip_start)


def format_clock(seconds: float) -> str:
    """``MM:SS`` from kickoff, which is how a match clock is read."""
    if seconds is None or np.isnan(seconds):
        return "--:--"
    total = round(float(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def event_surprisal(
    events: pd.DataFrame,
    tokenizer: Tokenizer,
    model: PhaseLM,
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Attach ``surprisal_bits`` to every event.

    The first event of a possession is conditioned on ``<bos>`` and nothing
    else, so its surprisal measures the corpus-wide prior over set pieces
    rather than anything about this match. Scoring it anyway fills the clip
    list with lineouts. It is left as ``NaN``, and everything downstream skips
    NaN explicitly.
    """
    import torch

    from ruck.lm.data import pick_device

    torch_device = pick_device(device)
    model.eval()
    model.to(torch_device)

    bits = pd.Series(np.nan, index=events.index, dtype=float, name="surprisal_bits")
    with torch.no_grad():
        for sequence in tokenizer.encode_events(events):
            width = model.config.max_len
            ids = torch.from_numpy(sequence.ids[:width]).unsqueeze(0).to(torch_device)
            row = model.token_surprisal(ids).squeeze(0).cpu().numpy()
            for position, event_index in enumerate(sequence.event_index[:width]):
                # Position 0 is <bos>; position 1 is the first real event, whose
                # only context is <bos>. Neither says anything about this match.
                if position <= 1:
                    continue
                if event_index >= 0 and event_index in bits.index:
                    bits.loc[event_index] = float(row[position])

    out = events.copy()
    out["surprisal_bits"] = bits
    out["zone"] = [
        zone_of(None if pd.isna(x) else float(x), None if pd.isna(y) else float(y))
        for x, y in zip(out["x"], out["y"], strict=True)
    ]
    return out


def phase_timeline(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean surprisal per (possession, phase), in match order.

    This is the series the timeline chart plots: one point per phase of play,
    so a 12-phase possession contributes 12 points rather than one.
    """
    usable = scored[scored["surprisal_bits"].notna()]
    if usable.empty:
        return pd.DataFrame(
            columns=[
                "possession_id",
                "phase",
                "t",
                "period",
                "team",
                "mean_bits",
                "max_bits",
                "n_events",
            ]
        )
    grouped = (
        usable.groupby(["match_id", "possession_id", "phase"], dropna=False)
        .agg(
            t=("t", "min"),
            period=("period", "first"),
            team=("team", "first"),
            mean_bits=("surprisal_bits", "mean"),
            max_bits=("surprisal_bits", "max"),
            n_events=("surprisal_bits", "size"),
        )
        .reset_index()
        .sort_values("t", kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def describe(row: pd.Series) -> str:
    """A one-line description built only from the token fields."""
    token = token_of(row)
    action = ACTION_PHRASE.get(token.action, token.action.replace("_", " "))
    outcome = OUTCOME_PHRASE.get(token.outcome, token.outcome.replace("_", " "))
    team = "" if pd.isna(row.get("team")) else str(row["team"])

    if token.zone == "??":
        where = "at an unrecorded position"
    else:
        band, channel = token.zone.split(":")
        where = f"{ZONE_PHRASE.get(band, band)}, {CHANNEL_PHRASE.get(channel, channel)}"

    phase = row.get("phase")
    if pd.isna(phase):
        when = ""
    elif int(phase) == 0:
        when = " off the set piece"
    else:
        when = f" on phase {int(phase)}"

    return f"{team} {action} {where}{when} — {outcome}".strip()


def top_sequences(
    scored: pd.DataFrame,
    *,
    n: int = 10,
    window: int = 3,
    min_gap: float = 15.0,
    max_span: float = 150.0,
) -> list[Highlight]:
    """The ``n`` most unexpected passages of play, as clip-ready highlights.

    A passage is a run of ``window`` consecutive events inside one possession.
    Windows are ranked by total surprisal and then thinned so that two
    highlights are never within ``min_gap`` seconds of each other, which stops
    the list from being ten cuts of the same try.
    """
    usable = scored[scored["surprisal_bits"].notna()].sort_values("t", kind="stable")
    if usable.empty:
        return []

    candidates: list[tuple[float, float, int, pd.DataFrame]] = []
    for _, block in usable.groupby(["match_id", "possession_id"], dropna=False):
        block = block.sort_values("t", kind="stable")
        values = block["surprisal_bits"].to_numpy()
        width = min(window, len(block))
        for start in range(len(block) - width + 1):
            chunk = block.iloc[start : start + width]
            # Three events of one passage of play cannot span two minutes. If
            # they appear to, the source's possession ids are not what they
            # claim and the clip would point at the wrong part of the video.
            if float(chunk["t"].iloc[-1] - chunk["t"].iloc[0]) > max_span:
                continue
            total = float(values[start : start + width].sum())
            peak_offset = int(np.argmax(values[start : start + width]))
            candidates.append((total, float(chunk["t"].iloc[0]), peak_offset, chunk))

    candidates.sort(key=lambda item: (-item[0], item[1]))

    highlights: list[Highlight] = []
    taken: list[float] = []
    for total, start_t, peak_offset, chunk in candidates:
        if len(highlights) >= n:
            break
        if any(abs(start_t - other) < min_gap for other in taken):
            continue
        peak = chunk.iloc[peak_offset]
        highlights.append(
            Highlight(
                rank=len(highlights) + 1,
                t=float(peak["t"]),
                clip_start=max(0.0, start_t - CLIP_LEAD_IN),
                period=int(peak["period"]) if pd.notna(peak["period"]) else 1,
                team="" if pd.isna(peak["team"]) else str(peak["team"]),
                phase=int(peak["phase"]) if pd.notna(peak["phase"]) else 0,
                total_bits=total,
                peak_bits=float(peak["surprisal_bits"]),
                n_events=len(chunk),
                description=describe(peak),
                event_index=int(peak.name),
            )
        )
        taken.append(start_t)
    return highlights


def summarise(scored: pd.DataFrame) -> dict[str, float]:
    """Headline surprisal numbers for the report cover."""
    usable = scored["surprisal_bits"].dropna()
    if usable.empty:
        return {
            "mean_bits": float("nan"),
            "median_bits": float("nan"),
            "p90_bits": float("nan"),
            "n_scored": 0.0,
        }
    return {
        "mean_bits": float(usable.mean()),
        "median_bits": float(usable.median()),
        "p90_bits": float(usable.quantile(0.9)),
        "n_scored": float(len(usable)),
    }
