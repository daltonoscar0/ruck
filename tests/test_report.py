from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from ruck.demo_data.generate import generate_corpus, generate_match
from ruck.report import charts
from ruck.report.render import ReportContext, render_report
from ruck.report.shape import (
    compare_to_reference,
    competition_reference,
    team_shape,
)
from ruck.report.surprisal import (
    Highlight,
    describe,
    format_clock,
    phase_timeline,
    top_sequences,
)
from ruck.report.timestamps import COLUMNS, highlight_rows
from ruck.report.timestamps import write_csv as write_clip_csv
from ruck.schema import coerce


@pytest.fixture(scope="module")
def match() -> pd.DataFrame:
    return generate_match(seed=42, home="Kestrels", away="Stags")


@pytest.fixture(scope="module")
def scored(match: pd.DataFrame) -> pd.DataFrame:
    """A match with a deterministic stand-in for model surprisal.

    The report layer is tested without a trained model: what matters here is
    that ranking, grouping and rendering behave, not what the numbers are.
    """
    out = match.copy()
    out["surprisal_bits"] = [float((index * 7919) % 100) / 10.0 for index in range(len(out))]
    out.loc[out.index[0], "surprisal_bits"] = float("nan")
    return out


# -- clock and descriptions ----------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0.0, "00:00"), (61.4, "01:01"), (2760.0, "46:00"), (4800.0, "80:00")],
)
def test_format_clock(seconds, text):
    assert format_clock(seconds) == text


def test_format_clock_handles_missing():
    assert format_clock(float("nan")) == "--:--"


def test_description_is_built_from_the_token_fields():
    row = coerce(
        pd.DataFrame(
            [
                {
                    "match_id": "m",
                    "t": 0.0,
                    "team": "Munster",
                    "type": "carry",
                    "x": 85.0,
                    "y": 60.0,
                    "phase": 3,
                    "outcome": "retained",
                }
            ]
        )
    ).iloc[0]
    text = describe(row)
    assert "Munster" in text
    assert "carry" in text
    assert "inside the opposition 22" in text
    assert "right channel" in text
    assert "phase 3" in text
    assert "ball retained" in text


def test_description_says_so_when_there_are_no_coordinates():
    row = coerce(
        pd.DataFrame(
            [
                {
                    "match_id": "m",
                    "t": 0.0,
                    "team": "Italy",
                    "type": "try",
                    "phase": 0,
                    "outcome": "points",
                }
            ]
        )
    ).iloc[0]
    assert "unrecorded position" in describe(row)


def test_set_piece_phase_reads_as_off_the_set_piece():
    row = coerce(
        pd.DataFrame(
            [
                {
                    "match_id": "m",
                    "t": 0.0,
                    "team": "A",
                    "type": "lineout",
                    "x": 40.0,
                    "y": 5.0,
                    "phase": 0,
                    "outcome": "won",
                }
            ]
        )
    ).iloc[0]
    assert "off the set piece" in describe(row)


# -- timeline ------------------------------------------------------------


def test_phase_timeline_has_one_row_per_phase(scored: pd.DataFrame):
    timeline = phase_timeline(scored)
    expected = (
        scored[scored["surprisal_bits"].notna()]
        .groupby(["match_id", "possession_id", "phase"])
        .ngroups
    )
    assert len(timeline) == expected
    assert timeline["t"].is_monotonic_increasing


def test_phase_timeline_does_not_merge_two_matches():
    corpus = generate_corpus(n_matches=3, seed=4)
    corpus["surprisal_bits"] = 1.0
    per_match = sum(len(phase_timeline(block)) for _, block in corpus.groupby("match_id"))
    assert len(phase_timeline(corpus)) == per_match


def test_phase_timeline_of_an_unscored_match_is_empty(match: pd.DataFrame):
    blank = match.copy()
    blank["surprisal_bits"] = float("nan")
    assert phase_timeline(blank).empty


# -- highlights ----------------------------------------------------------


def test_top_sequences_returns_the_requested_number(scored: pd.DataFrame):
    highlights = top_sequences(scored, n=10)
    assert len(highlights) == 10
    assert [h.rank for h in highlights] == list(range(1, 11))


def test_highlights_are_ordered_by_total_surprisal(scored: pd.DataFrame):
    totals = [h.total_bits for h in top_sequences(scored, n=10)]
    assert totals == sorted(totals, reverse=True)


def test_clip_starts_before_the_moment_it_shows(scored: pd.DataFrame):
    for highlight in top_sequences(scored, n=10):
        assert highlight.clip_start < highlight.t
        assert highlight.clip_start >= 0.0


def test_clips_are_spread_out_rather_than_ten_cuts_of_one_try(scored: pd.DataFrame):
    times = sorted(h.clip_start for h in top_sequences(scored, n=10, min_gap=15.0))
    gaps = [b - a for a, b in pairwise(times)]
    assert all(gap >= 15.0 for gap in gaps)


def test_a_window_spanning_an_implausible_stretch_of_play_is_rejected():
    frame = coerce(
        pd.DataFrame(
            [
                {
                    "match_id": "m",
                    "t": 10.0,
                    "team": "A",
                    "type": "carry",
                    "phase": 0,
                    "possession_id": 1,
                    "x": 50.0,
                    "y": 35.0,
                },
                {
                    "match_id": "m",
                    "t": 12.0,
                    "team": "A",
                    "type": "ruck",
                    "phase": 1,
                    "possession_id": 1,
                    "x": 52.0,
                    "y": 35.0,
                },
                # Same possession id, twenty minutes later: a broken source.
                {
                    "match_id": "m",
                    "t": 1300.0,
                    "team": "A",
                    "type": "carry",
                    "phase": 2,
                    "possession_id": 1,
                    "x": 54.0,
                    "y": 35.0,
                },
            ]
        )
    )
    frame["surprisal_bits"] = [5.0, 6.0, 7.0]
    assert top_sequences(frame, n=5, window=3, max_span=150.0) == []


def test_no_highlights_when_nothing_was_scored(match: pd.DataFrame):
    blank = match.copy()
    blank["surprisal_bits"] = float("nan")
    assert top_sequences(blank) == []


def test_highlight_rows_cover_every_csv_column(scored: pd.DataFrame):
    rows = highlight_rows(top_sequences(scored, n=3))
    assert rows
    for row in rows:
        assert set(row) == set(COLUMNS)


def test_clip_csv_is_written_with_a_bom_so_excel_reads_it(tmp_path, scored: pd.DataFrame):
    path = write_clip_csv(top_sequences(scored, n=4), tmp_path / "clips.csv")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == list(COLUMNS)
    assert len(reloaded) == 4


# -- attack shape --------------------------------------------------------


def test_team_shape_counts_only_that_team(match: pd.DataFrame):
    whole = team_shape(match)
    kestrels = team_shape(match, "Kestrels")
    stags = team_shape(match, "Stags")
    assert kestrels.carries + stags.carries == whole.carries
    assert kestrels.team == "Kestrels"


def test_possessions_are_not_merged_across_matches():
    corpus = generate_corpus(n_matches=4, seed=8)
    shape = team_shape(corpus)
    per_match = sum(team_shape(block).n_possessions for _, block in corpus.groupby("match_id"))
    assert shape.n_possessions == per_match
    # A merged corpus would report hundreds of carries in a single possession.
    assert shape.carries_per_possession < 12


def test_phase_histogram_is_a_distribution(match: pd.DataFrame):
    histogram = team_shape(match).phase_histogram
    assert pytest.approx(sum(histogram.values()), abs=1e-9) == 1.0


def test_metres_per_carry_uses_the_recorded_end_of_the_carry(match: pd.DataFrame):
    metres = team_shape(match).metres_per_carry
    assert metres is not None
    assert 0.5 < metres < 12.0


def test_shape_without_coordinates_reports_none_rather_than_zero(match: pd.DataFrame):
    blind = match.copy()
    blind[["x", "y", "end_x", "end_y"]] = None
    shape = team_shape(blind)
    assert shape.width_share is None
    assert shape.territory_share is None
    assert shape.carries > 0


def test_competition_reference_excludes_the_match_under_review():
    corpus = generate_corpus(n_matches=5, seed=6)
    match_id = sorted(corpus["match_id"].unique())[0]
    reference = competition_reference(corpus, exclude_match=match_id)
    only_that_match = team_shape(corpus[corpus["match_id"] == match_id])
    assert reference.n_events == len(corpus) - only_that_match.n_events
    assert reference.team == "Competition average"


def test_comparison_table_lines_metrics_up(match: pd.DataFrame):
    corpus = generate_corpus(n_matches=3, seed=2)
    table = compare_to_reference(team_shape(match), competition_reference(corpus))
    assert set(table.columns) == {"metric", "match", "competition", "delta"}
    assert not table.empty
    assert table["delta"].notna().any()


# -- charts and PDF ------------------------------------------------------


def test_every_chart_returns_a_png(scored: pd.DataFrame, match: pd.DataFrame):
    timeline = phase_timeline(scored)
    highlights = top_sequences(scored, n=5)
    shape = team_shape(match)
    images = [
        charts.timeline_chart(timeline, highlights),
        charts.phase_histogram_chart(shape.phase_histogram, None),
        charts.width_chart(shape.width_share, None),
        charts.territory_chart(shape.territory_share, None),
        charts.surprisal_distribution_chart(scored["surprisal_bits"].to_numpy(dtype=float)),
        charts.overlay_timeline_chart([("a", timeline), ("b", timeline)]),
    ]
    for png in images:
        assert png.startswith(b"\x89PNG")


def test_charts_cope_with_a_source_that_has_no_coordinates():
    assert charts.width_chart(None, None).startswith(b"\x89PNG")
    assert charts.territory_chart(None, None).startswith(b"\x89PNG")


def test_report_renders_within_the_page_budget(tmp_path, scored, match):
    path = render_report(
        scored=scored,
        timeline=phase_timeline(scored),
        highlights=top_sequences(scored, n=10),
        match_shape=team_shape(match),
        reference_shape=competition_reference(generate_corpus(n_matches=3, seed=3)),
        context=ReportContext(
            match_id="syn-test",
            competition="Synthetic Phase League (generated)",
            season="2024",
            teams=("Kestrels", "Stags"),
            source_label="csv:test",
            synthetic=True,
            model_card={"n_params": 1_000_000, "ppl_transformer": 4.3, "ppl_bigram": 11.0},
        ),
        out_path=tmp_path / "report.pdf",
    )
    raw = path.read_bytes()
    assert raw.startswith(b"%PDF")
    pages = raw.count(b"/Type /Page\n") or raw.count(b"/Type /Page")
    assert 4 <= pages <= 6, f"the spec asks for 4-6 pages, got {pages}"


def test_a_synthetic_report_says_so_on_the_cover(tmp_path, scored, match):
    path = render_report(
        scored=scored,
        timeline=phase_timeline(scored),
        highlights=top_sequences(scored, n=3),
        match_shape=team_shape(match),
        reference_shape=None,
        context=ReportContext(match_id="syn", synthetic=True, source_label="csv:test"),
        out_path=tmp_path / "synthetic.pdf",
    )
    assert b"SYNTHETIC DATA" in path.read_bytes()


def test_a_report_warns_when_the_model_loses_to_the_bigram(tmp_path, scored, match):
    path = render_report(
        scored=scored,
        timeline=phase_timeline(scored),
        highlights=top_sequences(scored, n=3),
        match_shape=team_shape(match),
        reference_shape=None,
        context=ReportContext(
            match_id="syn",
            source_label="csv:test",
            model_card={"ppl_transformer": 20.0, "ppl_bigram": 11.0},
        ),
        out_path=tmp_path / "weak.pdf",
    )
    assert b"does not beat a bigram baseline" in path.read_bytes()


def test_highlight_is_serialisable_to_a_row():
    highlight = Highlight(
        rank=1,
        t=90.0,
        clip_start=82.0,
        period=1,
        team="Munster",
        phase=2,
        total_bits=12.0,
        peak_bits=8.0,
        n_events=3,
        description="a carry",
        event_index=4,
    )
    row = highlight_rows([highlight])[0]
    assert row["clock"] == "01:30"
    assert row["clip_start_clock"] == "01:22"
    assert row["team"] == "Munster"
