from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ruck.ingest import load, load_csv, load_espn, load_rugbypy
from ruck.ingest.espn import LEAGUES, resolve_league
from ruck.ingest.generic_csv import resolve_columns
from ruck.ingest.http import Fetcher, OfflineCacheMiss
from ruck.ingest.rugbypy import NoEventDataError, match_context
from ruck.ingest.vocab import fold, map_action, map_outcome, parse_time

FIXTURES = Path(__file__).parent / "fixtures"
ESPN_CACHE = FIXTURES / "espn_cache"


# -- vocabulary folding --------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ball Carry", "ball_carry"),
        ("KNOCK-ON", "knock_on"),
        ("  pick & go ", "pick_go"),
        ("Cañas", "canas"),
        ("22m restart", "22m_restart"),
    ],
)
def test_fold_is_deterministic_and_punctuation_insensitive(raw, expected):
    assert fold(raw) == expected
    assert fold(raw) == fold(raw)


@pytest.mark.parametrize(
    ("raw", "action"),
    [
        ("Pick & Go", "carry"),
        ("Breakdown", "ruck"),
        ("box kick", "kick"),
        ("Line-Out", "lineout"),
        ("knock on", "knock_on"),
        ("Yellow Card", "card"),
        ("driving maul", "maul"),
    ],
)
def test_action_aliases_fold_onto_the_canonical_vocabulary(raw, action):
    assert map_action(raw)[0] == action


def test_unknown_action_becomes_other_and_keeps_its_label():
    action, subtype = map_action("Contestable Restart Regather")
    assert action == "other"
    assert subtype == "contestable_restart_regather"


def test_a_canonical_label_carries_no_redundant_subtype():
    assert map_action("carry") == ("carry", None)


@pytest.mark.parametrize(
    ("raw", "outcome"),
    [
        ("Successful", "complete"),
        ("quick ball", "retained"),
        ("Turnover Lost", "lost"),
        ("out", "into_touch"),
        (None, "unknown"),
        ("gibberish", "unknown"),
    ],
)
def test_outcome_aliases(raw, outcome):
    assert map_outcome(raw) == outcome


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [(12.5, 12.5), ("03:20", 200.0), ("01:03:20", 3800.0), ("46'", 2760.0), ("90", 90.0)],
)
def test_clock_parsing(raw, seconds):
    assert parse_time(raw) == pytest.approx(seconds)


def test_unparseable_clock_returns_none_rather_than_guessing():
    assert parse_time("second half") is None


# -- generic CSV ---------------------------------------------------------


def test_mini_match_loads_with_mixed_case_and_aliased_labels():
    events, report = load_csv(FIXTURES / "mini_match.csv", competition="URC", season="2024")
    assert report.events_out == 13
    assert not report.dropped
    assert events["type"].tolist()[7:] == ["lineout", "carry", "ruck", "carry", "ruck", "knock_on"]
    assert events["competition"].unique().tolist() == ["URC"]


def test_quick_ball_maps_to_retained():
    events, _ = load_csv(FIXTURES / "mini_match.csv")
    ruck = events[(events["team"] == "Leinster") & (events["type"] == "ruck")].iloc[0]
    assert ruck["outcome"] == "retained"


def test_column_alias_resolution():
    resolved = resolve_columns(["Time", "Squad", "Event Type", "Start X", "start_y", "Result"])
    assert resolved["t"] == "Time"
    assert resolved["team"] == "Squad"
    assert resolved["type"] == "Event Type"
    assert resolved["x"] == "Start X"
    assert resolved["outcome"] == "Result"


def test_missing_required_column_names_what_is_missing(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("t,x,y\n0,10,10\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_csv(path)


def test_percent_coordinates_are_rescaled(tmp_path):
    path = tmp_path / "pct.csv"
    path.write_text("t,team,action,x,y,phase,outcome\n0,A,carry,50,100,0,retained\n")
    events, report = load_csv(path)
    assert events["y"].iloc[0] == pytest.approx(70.0)
    assert any("percent" in note for note in report.notes)


def test_absolute_direction_mirrors_the_away_team(tmp_path):
    path = tmp_path / "abs.csv"
    path.write_text(
        "t,team,action,x,y,phase,outcome,period\n"
        "0,Home,carry,80,10,0,retained,1\n"
        "10,Away,carry,80,10,0,retained,1\n"
    )
    events, _ = load_csv(path, direction="absolute", home_team="Home")
    assert events.loc[events["team"] == "Home", "x"].iloc[0] == pytest.approx(80.0)
    assert events.loc[events["team"] == "Away", "x"].iloc[0] == pytest.approx(20.0)
    assert events.loc[events["team"] == "Away", "y"].iloc[0] == pytest.approx(60.0)


def test_absolute_direction_needs_a_home_team(tmp_path):
    path = tmp_path / "abs.csv"
    path.write_text("t,team,action,x,y,phase,outcome\n0,A,carry,80,10,0,retained\n")
    with pytest.raises(ValueError, match="home_team"):
        load_csv(path, direction="absolute")


def test_unmapped_actions_are_counted_not_dropped(tmp_path):
    path = tmp_path / "odd.csv"
    path.write_text(
        "t,team,action,x,y,phase,outcome\n"
        "0,A,carry,50,35,0,retained\n"
        "3,A,Choke Tackle Turnover,52,35,0,lost\n"
    )
    events, report = load_csv(path)
    assert len(events) == 2
    assert events["type"].tolist() == ["carry", "other"]
    assert report.unmapped_actions["choke_tackle_turnover"] == 1


def test_out_of_range_rows_are_dropped_and_reported(tmp_path):
    path = tmp_path / "wild.csv"
    path.write_text(
        "t,team,action,x,y,phase,outcome\n0,A,carry,50,35,0,retained\n3,A,carry,-40,35,0,retained\n"
    )
    events, report = load_csv(path)
    assert len(events) == 1
    assert len(report.dropped) == 1
    assert report.notes


def test_strict_mode_raises_instead_of_dropping(tmp_path):
    path = tmp_path / "wild.csv"
    path.write_text("t,team,action,x,y,phase,outcome\n0,A,carry,-40,35,0,retained\n")
    with pytest.raises(ValueError):
        load_csv(path, strict=True)


def test_a_file_with_no_coordinates_still_loads(tmp_path):
    path = tmp_path / "nocoords.csv"
    path.write_text("t,team,action,phase,outcome\n0,A,lineout,0,won\n4,A,carry,0,retained\n")
    events, report = load_csv(path)
    assert len(events) == 2
    assert events["x"].isna().all()
    assert any("no coordinates" in note for note in report.notes)


# -- ESPN ----------------------------------------------------------------


def test_league_slugs_and_raw_ids_both_resolve():
    assert resolve_league("six_nations") == LEAGUES["six_nations"] == "180659"
    assert resolve_league("267979") == "267979"
    with pytest.raises(ValueError, match="unknown league"):
        resolve_league("rugby league")


def test_espn_loads_offline_from_the_vendored_fixture():
    fetcher = Fetcher(cache_dir=ESPN_CACHE, offline=True)
    events, report = load_espn("597378", "six_nations", fetcher)
    assert report.rows_in == report.events_out == 49
    assert not report.dropped
    assert sorted(events["team"].dropna().unique()) == ["England", "Italy"]
    assert events["competition"].iloc[0] == "Six Nations"
    assert events["season"].iloc[0] == "2024"


def test_espn_resolves_player_names_from_the_roster():
    fetcher = Fetcher(cache_dir=ESPN_CACHE, offline=True)
    events, _ = load_espn("597378", "six_nations", fetcher)
    assert events["player"].notna().all()
    assert "Tommaso Allan" in set(events["player"])


def test_espn_events_carry_no_coordinates_and_say_so():
    fetcher = Fetcher(cache_dir=ESPN_CACHE, offline=True)
    events, report = load_espn("597378", "six_nations", fetcher)
    assert events["x"].isna().all() and events["y"].isna().all()
    assert any("no coordinates" in note for note in report.notes)


def test_espn_scoring_events_are_typed_and_ordered():
    fetcher = Fetcher(cache_dir=ESPN_CACHE, offline=True)
    events, _ = load_espn("597378", "six_nations", fetcher)
    counts = events["type"].value_counts().to_dict()
    assert counts["try"] == 5
    assert counts["penalty_goal"] == 6
    assert counts["conversion"] == 4
    assert counts["card"] == 2
    assert events["t"].is_monotonic_increasing


def test_offline_fetcher_raises_a_useful_error_on_a_cache_miss():
    fetcher = Fetcher(cache_dir=ESPN_CACHE, offline=True)
    with pytest.raises(OfflineCacheMiss, match="not cached"):
        load_espn("000000", "six_nations", fetcher)


def test_fetcher_default_user_agent_is_descriptive_not_a_browser():
    agent = Fetcher(cache_dir=ESPN_CACHE, offline=True).user_agent
    assert "ruck/" in agent
    assert "Mozilla" not in agent


# -- rugbypy -------------------------------------------------------------


def test_rugbypy_summary_frame_refuses_to_pretend_it_has_events(tmp_path):
    path = tmp_path / "team_stats.csv"
    pd.DataFrame(
        [
            {
                "team": "Munster",
                "match_id": "1",
                "tackles": 140,
                "rucks_won": 88,
                "metres_carried": 402,
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(NoEventDataError, match="team-stats"):
        load_rugbypy(path)


def test_rugbypy_event_shaped_frame_loads(tmp_path):
    path = tmp_path / "tagged.csv"
    pd.DataFrame(
        [
            {
                "t": 0,
                "team": "Munster",
                "action": "lineout",
                "x": 40,
                "y": 20,
                "phase": 0,
                "outcome": "won",
            }
        ]
    ).to_csv(path, index=False)
    events, report = load_rugbypy(path)
    assert len(events) == 1
    assert report.source.startswith("rugbypy:")


def test_rugbypy_match_context():
    frame = pd.DataFrame(
        [
            {
                "match_id": "9",
                "competition": "Six Nations",
                "season": "2024",
                "home_team": "Italy",
                "away_team": "England",
                "home_score": 24,
                "away_score": 27,
                "venue": "Stadio Olimpico",
                "date": "2024-02-03",
            }
        ]
    )
    context = match_context(frame)
    assert context.home_team == "Italy"
    assert context.away_score == 27


# -- dispatcher ----------------------------------------------------------


def test_load_sniffs_a_generic_csv():
    events, report = load(FIXTURES / "mini_match.csv")
    assert len(events) == 13
    assert report.source.startswith("csv:")


def test_load_sniffs_a_summary_csv_and_gives_the_useful_error(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        [{"team": "A", "match_id": "1", "tackles": 1, "rucks_won": 1, "metres_carried": 1}]
    ).to_csv(path, index=False)
    with pytest.raises(NoEventDataError):
        load(path)


def test_load_parses_an_espn_target():
    fetcher = Fetcher(cache_dir=ESPN_CACHE, offline=True)
    events, _ = load("espn:six_nations:597378", fetcher=fetcher)
    assert len(events) == 49


def test_load_rejects_a_malformed_espn_target():
    with pytest.raises(ValueError, match="espn target"):
        load("espn:a:b:c")


def test_an_event_csv_missing_one_column_says_which_one(tmp_path):
    """Not "this is a box score" — the user is one column away from working."""
    path = tmp_path / "nearly.csv"
    path.write_text("t,action,x,y,phase,outcome\n0,carry,40,12,0,retained\n")
    with pytest.raises(ValueError, match=r"missing required column\(s\) \['team'\]"):
        load(path)


def test_a_real_summary_frame_still_routes_to_the_rugbypy_message(tmp_path):
    path = tmp_path / "team_stats.csv"
    pd.DataFrame(
        [
            {
                "team": "Munster",
                "match_id": "1",
                "tackles": 140,
                "rucks_won": 88,
                "metres_carried": 402,
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(NoEventDataError, match="team-stats"):
        load(path)
