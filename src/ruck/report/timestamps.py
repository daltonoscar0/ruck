"""The timestamp list, which is what an analyst actually cuts video from.

The same rows feed the PDF table and the CSV, from one function, so the two can
never drift. ``clip_start`` is a few seconds before the passage begins, because
a clip that opens on the moment itself is useless for showing why it happened.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ruck.report.surprisal import Highlight, format_clock

#: CSV header, in order. Plain names and no merged cells: this file is opened in
#: Excel by people who are about to sort it.
COLUMNS: tuple[str, ...] = (
    "clip",
    "clock",
    "clip_start_clock",
    "t_seconds",
    "clip_start_seconds",
    "period",
    "minute",
    "team",
    "phase",
    "events_in_window",
    "surprisal_bits",
    "window_bits",
    "description",
)


def highlight_rows(highlights: list[Highlight]) -> list[dict[str, str]]:
    """Render highlights to the exact strings that go in both the CSV and the PDF."""
    rows: list[dict[str, str]] = []
    for highlight in highlights:
        rows.append(
            {
                "clip": str(highlight.rank),
                "clock": format_clock(highlight.t),
                "clip_start_clock": format_clock(highlight.clip_start),
                "t_seconds": f"{highlight.t:.1f}",
                "clip_start_seconds": f"{highlight.clip_start:.1f}",
                "period": str(highlight.period),
                "minute": str(int(highlight.t // 60) + 1),
                "team": highlight.team,
                "phase": str(highlight.phase),
                "events_in_window": str(highlight.n_events),
                "surprisal_bits": f"{highlight.peak_bits:.2f}",
                "window_bits": f"{highlight.total_bits:.2f}",
                "description": highlight.description,
            }
        )
    return rows


def write_csv(highlights: list[Highlight], path: str | Path) -> Path:
    """Write the clip list. UTF-8 with a BOM so Excel reads accents correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(highlight_rows(highlights))
    return path
