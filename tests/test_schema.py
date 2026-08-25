from __future__ import annotations

import json

import pandas as pd
import pytest

from ruck.schema import (
    ACTIONS,
    EVENT_COLUMNS,
    PITCH_LENGTH,
    RugbyEvent,
    coerce,
    derive_possessions,
    empty_events,
    events_to_frame,
    partition_valid,
    validate_frame,
)


def test_empty_frame_has_canonical_columns_in_order():
    assert list(empty_events().columns) == list(EVENT_COLUMNS)


def test_sportio_columns_come_first():
    # Downstream suite tools read the canonical block positionally.
    assert EVENT_COLUMNS[:19] == (
        "match_id", "competition", "season", "team_id", "team", "player_id", "player",
        "t", "period", "type", "subtype", "x", "y", "end_x", "end_y", "outcome",
        "pressure", "body_part", "extra",
    )


def test_event_round_trips_through_a_frame():
    events = [
        RugbyEvent(match_id="m1", t=0.0, type="restart", x=50.0, y=35.0, phase=0),
        RugbyEvent(match_id="m1", t=4.0, type="carry", x=52.0, y=30.0, phase=0,
                   outcome="retained", extra={"note": "off the top"}),
    ]
    frame = events_to_frame(events)
    assert len(frame) == 2
    assert frame["type"].tolist() == ["restart", "carry"]
    assert json.loads(frame["extra"].iloc[1])["note"] == "off the top"


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError, match="unknown action"):
        RugbyEvent(match_id="m", t=0.0, type="scrummage")


def test_unknown_outcome_is_rejected():
    with pytest.raises(ValueError, match="unknown outcome"):
        RugbyEvent(match_id="m", t=0.0, type="carry", outcome="quite good")


def test_coordinates_outside_the_pitch_are_rejected():
    with pytest.raises(ValueError):
        RugbyEvent(match_id="m", t=0.0, type="carry", x=PITCH_LENGTH + 1.0, y=10.0)


def test_partition_valid_keeps_good_rows_and_reports_bad_ones():
    frame = coerce(
        pd.DataFrame(
            [
                {"match_id": "m", "t": 1.0, "type": "carry", "x": 10.0, "y": 10.0},
                {"match_id": "m", "t": 2.0, "type": "carry", "x": 400.0, "y": 10.0},
            ]
        )
    )
    good, bad = partition_valid(frame)
    assert len(good) == 1
    assert len(bad) == 1 and bad[0][0] == 1


def test_validate_frame_raises_on_the_first_bad_row():
    frame = coerce(pd.DataFrame([{"match_id": "m", "t": 1.0, "type": "nope"}]))
    with pytest.raises(ValueError, match="failed schema validation"):
        validate_frame(frame)


def test_coerce_adds_missing_columns_and_drops_extras():
    frame = pd.DataFrame([{"match_id": "m", "t": 0.0, "type": "carry", "nonsense": 1}])
    out = coerce(frame)
    assert list(out.columns) == list(EVENT_COLUMNS)
    assert out["outcome"].isna().all()


def test_derive_possessions_counts_phases_from_the_set_piece():
    frame = coerce(
        pd.DataFrame(
            [
                {"match_id": "m", "t": 0.0, "team": "A", "type": "lineout"},
                {"match_id": "m", "t": 1.0, "team": "A", "type": "carry"},
                {"match_id": "m", "t": 2.0, "team": "A", "type": "ruck"},
                {"match_id": "m", "t": 3.0, "team": "A", "type": "carry"},
                {"match_id": "m", "t": 4.0, "team": "A", "type": "ruck"},
                {"match_id": "m", "t": 5.0, "team": "A", "type": "knock_on"},
                {"match_id": "m", "t": 6.0, "team": "B", "type": "scrum"},
            ]
        )
    )
    out = derive_possessions(frame)
    assert out["phase"].tolist() == [0, 0, 1, 1, 2, 2, 0]
    assert out["possession_id"].nunique() == 2


def test_derive_possessions_leaves_a_source_that_knows_its_own_phases_alone():
    frame = coerce(
        pd.DataFrame(
            [
                {"match_id": "m", "t": 0.0, "team": "A", "type": "lineout", "phase": 0,
                 "possession_id": 7},
                {"match_id": "m", "t": 1.0, "team": "A", "type": "ruck", "phase": 9,
                 "possession_id": 7},
            ]
        )
    )
    out = derive_possessions(frame)
    assert out["phase"].tolist() == [0, 9]
    assert out["possession_id"].tolist() == [7, 7]


def test_every_action_in_the_vocabulary_validates():
    for action in ACTIONS:
        RugbyEvent(match_id="m", t=0.0, type=action)
