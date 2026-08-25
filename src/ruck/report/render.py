"""PDF renderer.

Five pages: cover, surprisal timeline, the unexpected-plays clip list, attack
shape against a competition reference, and a model card. The clip-list page is
built from the same rows as the CSV, so the two cannot disagree.

Pages are written uncompressed. That costs a few kilobytes and buys the ability
to assert, in a test, that a timestamp printed in the CSV is really present in
the PDF.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ruck.report import charts
from ruck.report.shape import ShapeSummary, compare_to_reference
from ruck.report.surprisal import Highlight
from ruck.report.timestamps import highlight_rows

PAGE_WIDTH, PAGE_HEIGHT = A4
CONTENT_WIDTH = PAGE_WIDTH - 36 * mm

INK = colors.HexColor(charts.TEXT_PRIMARY)
INK_SOFT = colors.HexColor(charts.TEXT_SECONDARY)
RULE = colors.HexColor(charts.GRID)
ACCENT = colors.HexColor(charts.SERIES_1)
WARN_BG = colors.HexColor("#fdeee6")
WARN_INK = colors.HexColor("#8a3a12")


@dataclass
class ReportContext:
    """Everything the cover and the model card need to be truthful."""

    match_id: str
    competition: str | None = None
    season: str | None = None
    teams: tuple[str, ...] = ()
    focus_team: str | None = None
    source_label: str = "unknown source"
    #: Set when the events were produced by the generator rather than measured.
    synthetic: bool = False
    #: Free-text lines from the load report, printed verbatim on the cover.
    load_notes: list[str] = field(default_factory=list)
    #: Model provenance: parameter count, perplexity, baselines.
    model_card: dict[str, object] = field(default_factory=dict)
    generated_at: str = ""
    #: An optional real-data appendix: ``{"label", "note", "columns", "rows"}``.
    #: Kept in its own section so measured events are never shown next to
    #: generated ones as if they were the same thing.
    real_check: dict[str, object] | None = None

    def title(self) -> str:
        if len(self.teams) >= 2:
            return f"{self.teams[0]} v {self.teams[1]}"
        return self.match_id

    def subtitle(self) -> str:
        parts = [p for p in (self.competition, self.season) if p]
        return " · ".join(parts) if parts else self.match_id


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=INK_SOFT,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=INK,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.5,
            textColor=INK,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=INK_SOFT,
            spaceAfter=3,
        ),
        "banner": ParagraphStyle(
            "banner",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=WARN_INK,
        ),
    }


def _image(png: bytes, width: float = CONTENT_WIDTH) -> Image:
    image = Image(io.BytesIO(png))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return image


def _table(data: list[list[str]], widths: list[float], *, header: bool = True) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), INK_SOFT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, INK_SOFT),
        ]
    table.setStyle(TableStyle(style))
    return table


def _provenance_banner(context: ReportContext, styles: dict[str, ParagraphStyle]) -> Table:
    if context.synthetic:
        text = (
            "SYNTHETIC DATA — these events were produced by ruck's generator, not "
            "measured at a match. Every number in this report describes the pipeline, "
            "not rugby. Do not quote it as analysis."
        )
        background = WARN_BG
        style = styles["banner"]
    else:
        text = f"Source: {context.source_label}"
        background = colors.HexColor("#f2f5fa")
        style = styles["small"]
    table = Table([[Paragraph(text, style)]], colWidths=[CONTENT_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "--"
    return f"{number:,.{digits}f}"


def render_report(
    *,
    scored: pd.DataFrame,
    timeline: pd.DataFrame,
    highlights: list[Highlight],
    match_shape: ShapeSummary,
    reference_shape: ShapeSummary | None,
    context: ReportContext,
    out_path: str | Path,
) -> Path:
    """Write the analysis PDF and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    generated = context.generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    document = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"ruck — {context.title()}",
        author="ruck",
        subject="rugby phase-play surprisal report",
    )
    # Uncompressed pages so the acceptance test can find the clip timestamps.
    document.pageCompression = 0

    story: list[object] = []
    story += _cover_page(scored, highlights, match_shape, context, styles, generated)
    story.append(PageBreak())
    story += _timeline_page(scored, timeline, highlights, styles)
    story.append(PageBreak())
    story += _clips_page(highlights, styles)
    story.append(PageBreak())
    story += _shape_page(match_shape, reference_shape, styles)
    story.append(PageBreak())
    story += _method_page(context, styles)
    if context.real_check:
        story.append(PageBreak())
        story += _real_check_page(context, styles)

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out_path


def _real_check_page(context: ReportContext, styles: dict[str, ParagraphStyle]) -> list[object]:
    """An appendix showing a real, measured match read by the same ingest layer.

    Separate page, separate source label, no shared axes with anything above it.
    The point is to show the loaders work against a live feed, not to mix
    measured events into generated analysis.
    """
    check = context.real_check or {}
    story: list[object] = [
        Paragraph(str(check.get("label", "Real-data check")), styles["h2"]),
        Paragraph(str(check.get("note", "")), styles["body"]),
    ]
    columns = list(check.get("columns") or [])
    rows = list(check.get("rows") or [])
    if columns and rows:
        widths = [CONTENT_WIDTH / len(columns)] * len(columns)
        story.append(_table([columns] + [[str(cell) for cell in row] for row in rows], widths))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "These events are measured, and are shown only to demonstrate the ingest "
            "path. They are not scored by the model and contribute nothing to any "
            "number on the pages above.",
            styles["small"],
        )
    )
    return story


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(INK_SOFT)
    canvas.drawString(18 * mm, 10 * mm, "ruck 0.1.0")
    canvas.drawRightString(PAGE_WIDTH - 18 * mm, 10 * mm, f"page {document.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 13 * mm, PAGE_WIDTH - 18 * mm, 13 * mm)
    canvas.restoreState()


def _cover_page(
    scored: pd.DataFrame,
    highlights: list[Highlight],
    match_shape: ShapeSummary,
    context: ReportContext,
    styles: dict[str, ParagraphStyle],
    generated: str,
) -> list[object]:
    bits = scored["surprisal_bits"].dropna()
    focus = f" — {context.focus_team}" if context.focus_team else ""
    story: list[object] = [
        Paragraph(f"{context.title()}{focus}", styles["title"]),
        Paragraph(f"{context.subtitle()} · phase-play surprisal report", styles["subtitle"]),
        _provenance_banner(context, styles),
        Spacer(1, 8),
    ]

    summary = [
        ["Events analysed", _fmt(len(scored), 0)],
        ["Events scored by the model", _fmt(int(bits.size), 0)],
        ["Possessions", _fmt(match_shape.n_possessions, 0)],
        ["Mean surprisal", f"{_fmt(bits.mean())} bits" if bits.size else "--"],
        ["90th percentile surprisal", f"{_fmt(bits.quantile(0.9))} bits" if bits.size else "--"],
        ["Clips in the list", _fmt(len(highlights), 0)],
    ]
    story.append(
        _table([["", ""], *summary], [CONTENT_WIDTH * 0.55, CONTENT_WIDTH * 0.45], header=False)
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("What this report is", styles["h2"]))
    story.append(
        Paragraph(
            "A language model reads the match one event at a time and, at each step, "
            "says how likely the next event was given everything before it. Surprisal "
            "is the negative log of that probability, in bits: a routine ruck after a "
            "carry costs a fraction of a bit, an offload out of contact on the "
            "opposition line costs many. The timeline shows where the match stopped "
            "being predictable; the clip list turns those moments into timestamps.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Surprisal is not quality. An unexpected event may be brilliant or a "
            "mistake — the model has no view on that. It is a way of finding the "
            "twenty seconds of a match worth watching again, not a rating.",
            styles["body"],
        )
    )

    if context.load_notes:
        story.append(Paragraph("Ingest notes", styles["h3"]))
        for note in context.load_notes:
            story.append(Paragraph(f"• {note}", styles["small"]))

    story.append(Spacer(1, 6))
    story.append(
        Paragraph(f"Source: {context.source_label} · generated {generated}", styles["small"])
    )
    return story


def _timeline_page(
    scored: pd.DataFrame,
    timeline: pd.DataFrame,
    highlights: list[Highlight],
    styles: dict[str, ParagraphStyle],
) -> list[object]:
    story: list[object] = [
        Paragraph("Surprisal timeline", styles["h2"]),
        Paragraph(
            "One point per phase of play, in match order. Peaks are where the model "
            "was most wrong about what came next.",
            styles["body"],
        ),
        _image(charts.timeline_chart(timeline, highlights)),
        Spacer(1, 8),
        Paragraph("Distribution", styles["h3"]),
        Paragraph(
            "Most events are ordinary; the tail is what the clip list is drawn from.",
            styles["body"],
        ),
        _image(charts.surprisal_distribution_chart(scored["surprisal_bits"].to_numpy(dtype=float))),
    ]

    if not timeline.empty:
        busiest = timeline.nlargest(min(5, len(timeline)), "mean_bits")
        rows = [["Minute", "Team", "Phase", "Events", "Mean bits"]]
        for _, row in busiest.iterrows():
            rows.append(
                [
                    f"{row['t'] / 60:.0f}'",
                    "" if pd.isna(row["team"]) else str(row["team"]),
                    "--" if pd.isna(row["phase"]) else str(int(row["phase"])),
                    str(int(row["n_events"])),
                    _fmt(row["mean_bits"]),
                ]
            )
        story.append(Spacer(1, 8))
        story.append(Paragraph("Most unpredictable phases", styles["h3"]))
        story.append(
            _table(
                rows,
                [
                    CONTENT_WIDTH * 0.14,
                    CONTENT_WIDTH * 0.34,
                    CONTENT_WIDTH * 0.14,
                    CONTENT_WIDTH * 0.16,
                    CONTENT_WIDTH * 0.22,
                ],
            )
        )
    return story


def _clips_page(highlights: list[Highlight], styles: dict[str, ParagraphStyle]) -> list[object]:
    story: list[object] = [
        Paragraph("Unexpected plays", styles["h2"]),
        Paragraph(
            "The passages of play the model least expected, ranked by total surprisal "
            "across a three-event window. <b>Clip</b> is the suggested cut point, eight "
            "seconds before the passage starts, so the build-up is on screen. These rows "
            "are the same rows as the exported CSV.",
            styles["body"],
        ),
        Spacer(1, 4),
    ]
    if not highlights:
        story.append(
            Paragraph("No events could be scored, so there is no clip list.", styles["body"])
        )
        return story

    rows = [["#", "Clock", "Clip from", "Team", "Phase", "Bits", "What happened"]]
    for row in highlight_rows(highlights):
        rows.append(
            [
                row["clip"],
                row["clock"],
                row["clip_start_clock"],
                row["team"],
                row["phase"],
                row["surprisal_bits"],
                Paragraph(row["description"], styles["small"]),
            ]
        )
    story.append(
        _table(
            rows,
            [
                CONTENT_WIDTH * 0.05,
                CONTENT_WIDTH * 0.09,
                CONTENT_WIDTH * 0.10,
                CONTENT_WIDTH * 0.14,
                CONTENT_WIDTH * 0.07,
                CONTENT_WIDTH * 0.08,
                CONTENT_WIDTH * 0.47,
            ],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Descriptions are generated from the event's own fields — action, zone, "
            "phase, outcome — through a fixed phrase table. Nothing in a description "
            "is inferred.",
            styles["small"],
        )
    )
    return story


def _shape_page(
    match_shape: ShapeSummary,
    reference_shape: ShapeSummary | None,
    styles: dict[str, ParagraphStyle],
) -> list[object]:
    story: list[object] = [
        Paragraph("Attack shape", styles["h2"]),
        Paragraph(
            f"{match_shape.team}: {match_shape.carries:,} carries, {match_shape.passes:,} "
            f"passes, {match_shape.kicks:,} kicks across {match_shape.n_possessions:,} "
            "possessions."
            + (
                ""
                if reference_shape is None
                else f" Compared against {reference_shape.team.lower()}."
            ),
            styles["body"],
        ),
        _image(
            charts.phase_histogram_chart(
                match_shape.phase_histogram,
                reference_shape.phase_histogram if reference_shape else None,
                match_label=match_shape.team,
            )
        ),
        Spacer(1, 4),
        _image(
            charts.width_chart(
                match_shape.width_share,
                reference_shape.width_share if reference_shape else None,
                match_label=match_shape.team,
            )
        ),
        Spacer(1, 4),
        _image(
            charts.territory_chart(
                match_shape.territory_share,
                reference_shape.territory_share if reference_shape else None,
                match_label=match_shape.team,
            )
        ),
    ]

    if reference_shape is not None:
        table = compare_to_reference(match_shape, reference_shape)
        rows = [["Metric", match_shape.team, reference_shape.team, "Difference"]]
        for _, row in table.iterrows():
            rows.append(
                [
                    str(row["metric"]),
                    _fmt(row["match"]),
                    _fmt(row["competition"]),
                    _fmt(row["delta"]),
                ]
            )
        story.append(Spacer(1, 6))
        story.append(
            KeepTogether(
                [
                    Paragraph("Against the competition", styles["h3"]),
                    _table(
                        rows,
                        [
                            CONTENT_WIDTH * 0.40,
                            CONTENT_WIDTH * 0.20,
                            CONTENT_WIDTH * 0.20,
                            CONTENT_WIDTH * 0.20,
                        ],
                    ),
                ]
            )
        )
    if match_shape.width_share is None:
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "This source has no pitch coordinates, so the width and territory "
                "sections could not be computed. Phase length and the kick/carry mix "
                "do not need them.",
                styles["small"],
            )
        )
    return story


def _method_page(context: ReportContext, styles: dict[str, ParagraphStyle]) -> list[object]:
    card = context.model_card
    story: list[object] = [
        Paragraph("Model card", styles["h2"]),
        Paragraph(
            "The numbers behind the timeline. A perplexity figure means nothing on its "
            "own, so the count-based baselines fitted on the same training split are "
            "printed next to it. If the transformer is not better than the bigram, this "
            "table says so.",
            styles["body"],
        ),
    ]

    rows = [["Property", "Value"]]
    for label, key, digits in (
        ("Parameters", "n_params", 0),
        ("Vocabulary size", "vocab_size", 0),
        ("Training events", "train_events", 0),
        ("Training matches", "train_matches", 0),
        ("Test perplexity — transformer", "ppl_transformer", 3),
        ("Test perplexity — bigram baseline", "ppl_bigram", 3),
        ("Test perplexity — unigram baseline", "ppl_unigram", 3),
    ):
        if key in card:
            rows.append([label, _fmt(card[key], digits)])
    for key in ("downstream_gbm", "downstream_lm_probe", "downstream_base_rate"):
        if key in card:
            rows.append(
                [str(key).replace("downstream_", "Downstream — ").replace("_", " "), str(card[key])]
            )
    story.append(_table(rows, [CONTENT_WIDTH * 0.55, CONTENT_WIDTH * 0.45]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Caveats", styles["h2"]))
    for caveat in _caveats(context):
        story.append(Paragraph(f"• {caveat}", styles["body"]))
    return story


def _caveats(context: ReportContext) -> list[str]:
    caveats = [
        "Surprisal is measured against this model, trained on this corpus. A different "
        "corpus produces different numbers; the ranking within one match is the part "
        "worth reading.",
        "The first event of a possession has no context inside it and is not scored.",
        "Tokens are coarse by design — action, an 18-cell zone, a phase bucket and an "
        "outcome. Two events that differ only in who carried are the same token.",
        "The model has no view on whether an unexpected event was good.",
    ]
    if context.synthetic:
        caveats.insert(
            0,
            "The events in this report were generated, not measured. Nothing here is a "
            "statement about any real match, team or player.",
        )
    if context.model_card.get("ppl_bigram") and context.model_card.get("ppl_transformer"):
        try:
            if float(context.model_card["ppl_transformer"]) > float(
                context.model_card["ppl_bigram"]
            ):
                caveats.insert(
                    0,
                    "On this corpus the transformer does not beat a bigram baseline. Its "
                    "surprisal is still usable for ranking moments within a match, but "
                    "the model is not learning long-range structure a count model misses.",
                )
        except (TypeError, ValueError):
            pass
    return caveats


def render_comparison(
    *,
    series: list[tuple[str, pd.DataFrame]],
    shapes: list[ShapeSummary],
    contexts: list[ReportContext],
    out_path: str | Path,
) -> Path:
    """Write the two-match surprisal overlay produced by `ruck compare`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    document = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="ruck — comparison",
        author="ruck",
        subject="rugby phase-play surprisal comparison",
    )
    document.pageCompression = 0

    names = " vs ".join(label for label, _ in series)
    story: list[object] = [
        Paragraph(names, styles["title"]),
        Paragraph("Phase-play surprisal overlay", styles["subtitle"]),
    ]
    if any(context.synthetic for context in contexts):
        story.append(_provenance_banner(contexts[0], styles))
        story.append(Spacer(1, 8))

    story += [
        Paragraph(
            "Both matches scored by the same model, so the two curves are on the same "
            "scale. A curve that sits higher throughout is a match the model found harder "
            "to anticipate, not necessarily a better one.",
            styles["body"],
        ),
        _image(charts.overlay_timeline_chart(series)),
        Spacer(1, 10),
        Paragraph("Attack shape side by side", styles["h2"]),
    ]

    if shapes:
        metrics = [metric for metric, _ in shapes[0].as_rows()]
        rows = [["Metric"] + [shape.team for shape in shapes]]
        lookup = [dict(shape.as_rows()) for shape in shapes]
        for metric in metrics:
            rows.append([metric] + [_fmt(entry.get(metric)) for entry in lookup])
        width = CONTENT_WIDTH * 0.44
        each = (CONTENT_WIDTH - width) / max(len(shapes), 1)
        story.append(_table(rows, [width] + [each] * len(shapes)))

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out_path
