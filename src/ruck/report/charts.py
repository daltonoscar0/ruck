"""Charts for the PDF, rendered to PNG bytes.

Print-only, so there is one light surface and no interaction layer. The
categorical slots are the first two of the reference palette (blue, orange) and
were run through the palette validator: worst all-pairs CVD separation 24.7
(protan), normal-vision 33.6, both well clear of the floors.

Chart rules that hold everywhere here: one y-axis per plot, solid hairline
grid, thin marks, a legend whenever two series share a plot, direct labels only
on the points that matter.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ruck.report.shape import PHASE_BUCKET_ORDER
from ruck.report.surprisal import Highlight
from ruck.tokenizer import X_LABELS, Y_LABELS

# The PDF page is white, so the chart surface is too: an off-white panel
# shows its edges against the page.
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
AXIS = "#c9c8c3"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

BAND_LABELS = {
    "own22": "Own 22",
    "own40": "Own half",
    "mid": "Middle",
    "opp40": "Opp half",
    "opp22": "Opp 22",
    "opp5": "Opp 5m",
}
CHANNEL_LABELS = {"L": "Left", "M": "Midfield", "R": "Right"}


def _figure(width: float = 6.4, height: float = 2.7):
    figure, axes = plt.subplots(figsize=(width, height), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    axes.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.6)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=7, length=3, width=0.6)
    return figure, axes


def _to_png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.tight_layout(pad=0.6)
    figure.savefig(buffer, format="png", facecolor=SURFACE)
    plt.close(figure)
    return buffer.getvalue()


def timeline_chart(timeline: pd.DataFrame, highlights: list[Highlight]) -> bytes:
    """Mean surprisal per phase across the match, with the clip list marked."""
    figure, axes = _figure(height=2.9)
    if timeline.empty:
        axes.text(
            0.5,
            0.5,
            "no scored phases",
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=9,
            transform=axes.transAxes,
        )
        return _to_png(figure)

    minutes = timeline["t"].to_numpy(dtype=float) / 60.0
    bits = timeline["mean_bits"].to_numpy(dtype=float)
    axes.plot(
        minutes, bits, color=SERIES_1, linewidth=1.2, label="Mean surprisal per phase", zorder=2
    )

    if highlights:
        clip_minutes = [h.t / 60.0 for h in highlights]
        clip_bits = [h.peak_bits for h in highlights]
        axes.scatter(
            clip_minutes,
            clip_bits,
            s=34,
            color=SERIES_2,
            zorder=4,
            edgecolor=SURFACE,
            linewidth=1.4,
            label="Unexpected plays (clip list)",
        )
        # Direct-label only the three most surprising, and only where the label
        # will not land on top of another one. The axis carries the rest.
        labelled: list[float] = []
        for highlight in sorted(highlights, key=lambda h: -h.peak_bits):
            if len(labelled) >= 3:
                break
            minute = highlight.t / 60.0
            if any(abs(minute - other) < 5.0 for other in labelled):
                continue
            axes.annotate(
                f"#{highlight.rank}",
                (minute, highlight.peak_bits),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=TEXT_PRIMARY,
            )
            labelled.append(minute)

    axes.axvline(40, color=AXIS, linewidth=0.6, zorder=1)
    # Along the foot, where it cannot collide with a labelled peak.
    axes.annotate(
        "half time",
        (40, axes.get_ylim()[0]),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=6.5,
        color=TEXT_SECONDARY,
    )
    axes.set_xlabel("Minute", fontsize=7.5, color=TEXT_SECONDARY)
    axes.set_ylabel("Surprisal (bits)", fontsize=7.5, color=TEXT_SECONDARY)
    axes.set_xlim(0, max(85.0, float(minutes.max()) + 2))
    axes.legend(
        frameon=False, fontsize=7, loc="upper left", labelcolor=TEXT_SECONDARY, handlelength=1.4
    )
    return _to_png(figure)


def _grouped_bars(
    axes,
    categories: list[str],
    match_values: list[float],
    reference_values: list[float] | None,
    match_label: str,
    reference_label: str,
) -> None:
    positions = np.arange(len(categories), dtype=float)
    if reference_values is None:
        axes.bar(positions, match_values, width=0.55, color=SERIES_1, label=match_label)
    else:
        # A 2px surface gap between adjacent bars rather than a border around them.
        width = 0.36
        gap = 0.02
        axes.bar(
            positions - (width + gap) / 2,
            match_values,
            width=width,
            color=SERIES_1,
            label=match_label,
        )
        axes.bar(
            positions + (width + gap) / 2,
            reference_values,
            width=width,
            color=SERIES_2,
            label=reference_label,
        )
        axes.legend(frameon=False, fontsize=7, labelcolor=TEXT_SECONDARY, handlelength=1.2)
    axes.set_xticks(positions)
    axes.set_xticklabels(categories, fontsize=7, color=TEXT_SECONDARY)


def phase_histogram_chart(
    match_histogram: dict[str, float],
    reference_histogram: dict[str, float] | None,
    *,
    match_label: str = "This match",
    reference_label: str = "Competition",
) -> bytes:
    """How long possessions lasted, in phase-count buckets."""
    figure, axes = _figure(height=1.85)
    categories = list(PHASE_BUCKET_ORDER)
    match_values = [100.0 * match_histogram.get(bucket, 0.0) for bucket in categories]
    reference_values = (
        [100.0 * reference_histogram.get(bucket, 0.0) for bucket in categories]
        if reference_histogram
        else None
    )
    _grouped_bars(axes, categories, match_values, reference_values, match_label, reference_label)
    axes.set_xlabel("Phases in possession", fontsize=7.5, color=TEXT_SECONDARY)
    axes.set_ylabel("Share of possessions (%)", fontsize=7.5, color=TEXT_SECONDARY)
    return _to_png(figure)


def width_chart(
    match_width: dict[str, float] | None,
    reference_width: dict[str, float] | None,
    *,
    match_label: str = "This match",
    reference_label: str = "Competition",
) -> bytes:
    """Where carries were taken across the pitch."""
    figure, axes = _figure(height=1.85)
    if not match_width:
        axes.text(
            0.5,
            0.5,
            "no coordinates in this source",
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=9,
            transform=axes.transAxes,
        )
        axes.set_axis_off()
        return _to_png(figure)
    categories = [CHANNEL_LABELS[c] for c in Y_LABELS]
    match_values = [100.0 * match_width.get(c, 0.0) for c in Y_LABELS]
    reference_values = (
        [100.0 * reference_width.get(c, 0.0) for c in Y_LABELS] if reference_width else None
    )
    _grouped_bars(axes, categories, match_values, reference_values, match_label, reference_label)
    axes.set_ylabel("Share of carries (%)", fontsize=7.5, color=TEXT_SECONDARY)
    return _to_png(figure)


def territory_chart(
    match_territory: dict[str, float] | None,
    reference_territory: dict[str, float] | None,
    *,
    match_label: str = "This match",
    reference_label: str = "Competition",
) -> bytes:
    """Where on the field carries happened, from own line to opposition line."""
    figure, axes = _figure(height=1.85)
    if not match_territory:
        axes.text(
            0.5,
            0.5,
            "no coordinates in this source",
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=9,
            transform=axes.transAxes,
        )
        axes.set_axis_off()
        return _to_png(figure)
    categories = [BAND_LABELS[b] for b in X_LABELS]
    match_values = [100.0 * match_territory.get(b, 0.0) for b in X_LABELS]
    reference_values = (
        [100.0 * reference_territory.get(b, 0.0) for b in X_LABELS] if reference_territory else None
    )
    _grouped_bars(axes, categories, match_values, reference_values, match_label, reference_label)
    axes.set_ylabel("Share of carries (%)", fontsize=7.5, color=TEXT_SECONDARY)
    return _to_png(figure)


def surprisal_distribution_chart(bits: np.ndarray) -> bytes:
    """Distribution of per-event surprisal, with the clip threshold marked."""
    figure, axes = _figure(height=2.2)
    values = bits[np.isfinite(bits)]
    if values.size == 0:
        axes.text(
            0.5,
            0.5,
            "nothing scored",
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=9,
            transform=axes.transAxes,
        )
        axes.set_axis_off()
        return _to_png(figure)
    axes.hist(values, bins=36, color=SERIES_1)
    p90 = float(np.quantile(values, 0.9))
    axes.axvline(p90, color=SERIES_2, linewidth=1.4)
    axes.annotate(
        f"90th pct  {p90:.1f} bits",
        (p90, axes.get_ylim()[1]),
        textcoords="offset points",
        xytext=(5, -10),
        fontsize=7,
        color=TEXT_PRIMARY,
    )
    axes.set_xlabel("Surprisal (bits)", fontsize=7.5, color=TEXT_SECONDARY)
    axes.set_ylabel("Events", fontsize=7.5, color=TEXT_SECONDARY)
    return _to_png(figure)


def overlay_timeline_chart(
    series: list[tuple[str, pd.DataFrame]],
) -> bytes:
    """Two matches' phase surprisal on one axis, for `ruck compare`."""
    figure, axes = _figure(height=2.9)
    colors = (SERIES_1, SERIES_2)
    plotted = False
    for (label, timeline), color in zip(series[:2], colors, strict=False):
        if timeline.empty:
            continue
        axes.plot(
            timeline["t"].to_numpy(dtype=float) / 60.0,
            timeline["mean_bits"].to_numpy(dtype=float),
            color=color,
            linewidth=1.2,
            label=label,
        )
        plotted = True
    if not plotted:
        axes.text(
            0.5,
            0.5,
            "no scored phases",
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=9,
            transform=axes.transAxes,
        )
        axes.set_axis_off()
        return _to_png(figure)
    axes.axvline(40, color=AXIS, linewidth=0.6)
    axes.set_xlabel("Minute", fontsize=7.5, color=TEXT_SECONDARY)
    axes.set_ylabel("Surprisal (bits)", fontsize=7.5, color=TEXT_SECONDARY)
    axes.set_xlim(0, 85)
    axes.legend(
        frameon=False, fontsize=7, loc="upper left", labelcolor=TEXT_SECONDARY, handlelength=1.4
    )
    return _to_png(figure)
