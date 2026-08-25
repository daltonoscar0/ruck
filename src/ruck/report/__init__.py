"""Surprisal analysis, attack shape, and the PDF/CSV they produce."""

from ruck.report.charts import overlay_timeline_chart
from ruck.report.render import ReportContext, render_report
from ruck.report.shape import (
    ShapeSummary,
    compare_to_reference,
    competition_reference,
    team_shape,
)
from ruck.report.surprisal import (
    Highlight,
    describe,
    event_surprisal,
    format_clock,
    phase_timeline,
    summarise,
    top_sequences,
)
from ruck.report.timestamps import COLUMNS, highlight_rows, write_csv

__all__ = [
    "COLUMNS",
    "Highlight",
    "ReportContext",
    "ShapeSummary",
    "compare_to_reference",
    "competition_reference",
    "describe",
    "event_surprisal",
    "format_clock",
    "highlight_rows",
    "overlay_timeline_chart",
    "phase_timeline",
    "render_report",
    "summarise",
    "team_shape",
    "top_sequences",
    "write_csv",
]
