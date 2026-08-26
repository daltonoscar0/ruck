"""Attack-shape summary: width, phase length, and the kick/carry mix.

Three questions a coach asks about a performance, each answered against a
competition reference so the number has something to sit next to. Where a
source has no coordinates the width and territory sections report ``None`` and
the report says which parts it could not compute, rather than printing a zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ruck.tokenizer import X_LABELS, Y_LABELS, phase_bucket, zone_of

BALL_IN_PLAY: frozenset[str] = frozenset({"carry", "pass", "offload", "kick", "ruck", "maul"})
PHASE_BUCKET_ORDER: tuple[str, ...] = ("0", "1", "2", "3", "4-5", "6-8", "9+")


@dataclass
class ShapeSummary:
    """How a team played, in the handful of numbers that fit on a page."""

    team: str
    n_possessions: int
    n_events: int
    carries: int
    passes: int
    kicks: int
    rucks: int
    #: Share of carries in each lateral channel; ``None`` without coordinates.
    width_share: dict[str, float] | None = None
    mean_width_offset: float | None = None
    #: Share of possessions reaching each phase-count bucket.
    phase_histogram: dict[str, float] = field(default_factory=dict)
    mean_phases: float = float("nan")
    kicks_per_100_carries: float = float("nan")
    kick_share_of_ball_in_play: float = float("nan")
    passes_per_carry: float = float("nan")
    carries_per_possession: float = float("nan")
    #: Share of carries in each longitudinal band; ``None`` without coordinates.
    territory_share: dict[str, float] | None = None
    metres_per_carry: float | None = None

    def as_rows(self) -> list[tuple[str, float | None]]:
        """Flat metric rows, for tabling against a reference."""
        rows: list[tuple[str, float | None]] = [
            (
                "Carries per possession",
                self.carries_per_possession if self.carries else None,
            ),
            ("Passes per carry", self.passes_per_carry),
            ("Kicks per 100 carries", self.kicks_per_100_carries),
            ("Kick share of ball-in-play", self.kick_share_of_ball_in_play),
            ("Mean phases per possession", self.mean_phases),
        ]
        if self.metres_per_carry is not None:
            rows.append(("Metres per carry", self.metres_per_carry))
        if self.mean_width_offset is not None:
            rows.append(("Mean distance from midfield (m)", self.mean_width_offset))
        if self.width_share:
            for channel in Y_LABELS:
                share = self.width_share.get(channel)
                rows.append(
                    (
                        f"Carry share, {channel} channel (%)",
                        None if share is None else 100.0 * share,
                    )
                )
        return rows


def team_shape(events: pd.DataFrame, team: str | None = None) -> ShapeSummary:
    """Summarise one team's attacking shape (or every team, if ``team`` is None)."""
    frame = events if team is None else events[events["team"] == team]
    label = team or "All teams"
    if frame.empty:
        return ShapeSummary(
            team=label, n_possessions=0, n_events=0, carries=0, passes=0, kicks=0, rucks=0
        )

    carries = frame[frame["type"] == "carry"]
    n_carries = len(carries)
    n_passes = int((frame["type"].isin(["pass", "offload"])).sum())
    n_kicks = int((frame["type"] == "kick").sum())
    n_rucks = int((frame["type"].isin(["ruck", "maul"])).sum())
    n_ball_in_play = int(frame["type"].isin(BALL_IN_PLAY).sum())

    # Group by match as well as possession: possession ids restart in every
    # match, so grouping on the id alone silently merges a whole competition
    # into one enormous possession.
    possessions = frame.groupby(["match_id", "possession_id"], dropna=False)
    n_possessions = len(possessions)
    max_phase = possessions["phase"].max().dropna()

    histogram = {bucket: 0.0 for bucket in PHASE_BUCKET_ORDER}
    for value in max_phase:
        histogram[phase_bucket(value)] = histogram.get(phase_bucket(value), 0.0) + 1.0
    if len(max_phase):
        histogram = {k: v / len(max_phase) for k, v in histogram.items()}

    summary = ShapeSummary(
        team=label,
        n_possessions=n_possessions,
        n_events=len(frame),
        carries=n_carries,
        passes=n_passes,
        kicks=n_kicks,
        rucks=n_rucks,
        phase_histogram=histogram,
        mean_phases=float(max_phase.mean()) if len(max_phase) else float("nan"),
        kicks_per_100_carries=100.0 * n_kicks / n_carries if n_carries else float("nan"),
        kick_share_of_ball_in_play=n_kicks / n_ball_in_play if n_ball_in_play else float("nan"),
        passes_per_carry=n_passes / n_carries if n_carries else float("nan"),
        carries_per_possession=n_carries / n_possessions if n_possessions else float("nan"),
    )

    # Territory and metres gained need x only. A source that records how far up
    # the field a carry was but not how wide should still get those sections,
    # rather than a page saying it has no coordinates at all.
    along = carries[carries["x"].notna()]
    if not along.empty:
        bands = [zone_of(float(x), 35.0).split(":")[0] for x in along["x"]]
        summary.territory_share = {band: float(bands.count(band)) / len(bands) for band in X_LABELS}
        summary.metres_per_carry = _metres_per_carry(frame)

    across = carries[carries["x"].notna() & carries["y"].notna()]
    if not across.empty:
        channels = [
            zone_of(float(x), float(y)).split(":")[1]
            for x, y in zip(across["x"], across["y"], strict=True)
        ]
        summary.width_share = {
            channel: float(channels.count(channel)) / len(channels) for channel in Y_LABELS
        }
        summary.mean_width_offset = float(np.abs(across["y"].astype(float) - 35.0).mean())
    return summary


def _metres_per_carry(frame: pd.DataFrame) -> float | None:
    """Mean metres gained per carry, measured from the next event's position."""
    carries = frame[frame["type"] == "carry"]
    # A source that records where a carry finished is the reliable case.
    with_end = carries[carries["x"].notna() & carries["end_x"].notna()]
    if len(with_end):
        return float((with_end["end_x"].astype(float) - with_end["x"].astype(float)).mean())

    # Otherwise infer the gain from where the next event happened.
    gains: list[float] = []
    for _, block in frame.groupby(["match_id", "possession_id"], dropna=False):
        block = block.sort_values("t", kind="stable")
        x = block["x"].to_numpy(dtype=float)
        types = block["type"].astype(str).to_numpy()
        for i in range(len(block) - 1):
            if types[i] == "carry" and np.isfinite(x[i]) and np.isfinite(x[i + 1]):
                gains.append(x[i + 1] - x[i])
    return float(np.mean(gains)) if gains else None


def compare_to_reference(match: ShapeSummary, reference: ShapeSummary) -> pd.DataFrame:
    """Table this match's shape against a competition reference."""
    reference_rows = dict(reference.as_rows())
    rows = []
    for metric, value in match.as_rows():
        baseline = reference_rows.get(metric)
        delta = (
            value - baseline
            if value is not None
            and baseline is not None
            and np.isfinite(value)
            and np.isfinite(baseline)
            else None
        )
        rows.append({"metric": metric, "match": value, "competition": baseline, "delta": delta})
    return pd.DataFrame(rows)


def competition_reference(
    corpus: pd.DataFrame,
    *,
    exclude_matches: set[str] | str | None = None,
) -> ShapeSummary:
    """Shape of the rest of the competition, with the match under review removed.

    Takes a set, because a frame handed to `analyze` can hold more than one
    match and leaving the others in would benchmark them against themselves.
    """
    if exclude_matches is None:
        frame = corpus
    else:
        excluded = {exclude_matches} if isinstance(exclude_matches, str) else set(exclude_matches)
        frame = corpus[~corpus["match_id"].astype(str).isin(excluded)]
    summary = team_shape(frame)
    summary.team = "Competition average"
    return summary
