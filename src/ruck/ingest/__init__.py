"""Ingest layer: every source in, one canonical events frame out.

Three real paths:

``generic_csv``
    An analyst's own export (``t, team, action, x, y, phase, outcome``). The
    only route to real phase-play data with coordinates.
``espn``
    ESPN's unofficial rugby endpoints. Scoring, cards and substitutions with a
    match clock; no coordinates.
``rugbypy``
    RugbyPy-format frames. Summary totals, used for context.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ruck.ingest.espn import LEAGUES, list_matches, load_espn
from ruck.ingest.generic_csv import LoadReport, load_csv
from ruck.ingest.http import Fetcher, OfflineCacheMiss
from ruck.ingest.rugbypy import NoEventDataError, load_rugbypy, looks_like_events

__all__ = [
    "LEAGUES",
    "Fetcher",
    "LoadReport",
    "NoEventDataError",
    "OfflineCacheMiss",
    "list_matches",
    "load",
    "load_csv",
    "load_espn",
    "load_rugbypy",
]


def load(
    target: str | Path,
    *,
    source: str = "auto",
    fetcher: Fetcher | None = None,
    **kwargs: object,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load ``target`` with the right loader.

    ``target`` is a path to a CSV or RugbyPy file, or ``espn:<event_id>`` /
    ``espn:<league>:<event_id>`` for a live or cached ESPN match.
    """
    text = str(target)
    if source == "auto":
        source = _sniff(text)

    if source == "espn":
        parts = text.split(":")
        if len(parts) == 2:
            _, event_id = parts
            league = "six_nations"
        elif len(parts) == 3:
            _, league, event_id = parts
        else:
            raise ValueError("espn target must be espn:<event_id> or espn:<league>:<event_id>")
        return load_espn(event_id, league, fetcher, **kwargs)  # type: ignore[arg-type]

    if source == "rugbypy":
        return load_rugbypy(target, **kwargs)  # type: ignore[arg-type]

    if source == "csv":
        return load_csv(target, **kwargs)  # type: ignore[arg-type]

    raise ValueError(f"unknown source {source!r}")


def _sniff(text: str) -> str:
    if text.startswith("espn:"):
        return "espn"
    path = Path(text)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return "rugbypy"
    if path.exists() and path.suffix.lower() == ".csv":
        # A rugbypy summary export is still a CSV; let the rugbypy loader own it
        # so the error message is the useful one.
        try:
            head = pd.read_csv(path, nrows=5)
        except Exception:
            return "csv"
        if not looks_like_events(head):
            return "rugbypy"
    return "csv"
