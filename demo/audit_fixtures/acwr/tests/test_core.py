import math

import pytest

from acwr import Zone, classify, ewma_acwr, monotony, rolling_acwr, strain


# ----------------------------------------------------------------- rolling


def test_steady_load_gives_ratio_of_one():
    points = rolling_acwr([100.0] * 40)
    assert points[-1].ratio == pytest.approx(1.0)
    assert points[-1].zone is Zone.OPTIMAL


def test_spike_pushes_ratio_up():
    # Three steady weeks, then a week at triple the load.
    # acute = 900, chronic = (300*21 + 900*7) / 28 = 450  ->  ratio 2.0
    points = rolling_acwr([300.0] * 21 + [900.0] * 7)
    assert points[-1].ratio == pytest.approx(2.0)
    assert points[-1].zone is Zone.HIGH_RISK


def test_single_hard_session_barely_moves_the_ratio():
    # A common misreading: one big day is not a spike in ACWR terms.
    points = rolling_acwr([300.0] * 27 + [900.0])
    assert points[-1].ratio == pytest.approx(1.2, abs=0.01)
    assert points[-1].zone is Zone.OPTIMAL


def test_detraining_pushes_ratio_down():
    points = rolling_acwr([500.0] * 21 + [0.0] * 7)
    assert points[-1].ratio < 0.8
    assert points[-1].zone is Zone.UNDERLOADED


def test_ratio_is_none_until_chronic_window_is_full():
    points = rolling_acwr([100.0] * 30)
    assert all(p.ratio is None for p in points[:27])
    assert points[27].ratio is not None
    assert not points[0].is_ready
    assert points[-1].is_ready


def test_partial_window_allowed_when_opted_in():
    points = rolling_acwr([100.0] * 10, require_full_window=False)
    assert points[-1].ratio == pytest.approx(1.0)


def test_all_zero_load_is_undefined_not_infinite():
    points = rolling_acwr([0.0] * 30, require_full_window=False)
    assert points[-1].ratio is None
    assert points[-1].zone is Zone.UNDEFINED


def test_windows_are_computed_correctly():
    loads = [float(i) for i in range(1, 29)]  # 1..28
    p = rolling_acwr(loads, acute=7, chronic=28)[-1]
    assert p.acute == pytest.approx(sum(range(22, 29)) / 7)
    assert p.chronic == pytest.approx(sum(range(1, 29)) / 28)


def test_output_length_matches_input():
    assert len(rolling_acwr([50.0] * 13)) == 13


def test_empty_input():
    assert rolling_acwr([]) == []
    assert ewma_acwr([]) == []


# -------------------------------------------------------------------- ewma


def test_ewma_steady_load_gives_ratio_of_one():
    points = ewma_acwr([200.0] * 60)
    assert points[-1].ratio == pytest.approx(1.0, abs=1e-9)


def test_ewma_reacts_faster_than_rolling_to_a_spike():
    loads = [300.0] * 27 + [1200.0]
    r = rolling_acwr(loads)[-1].ratio
    e = ewma_acwr(loads)[-1].ratio
    assert e > r


def test_ewma_seeds_from_first_day():
    points = ewma_acwr([100.0] + [0.0] * 30, require_full_window=False)
    assert points[0].acute == pytest.approx(100.0)
    assert points[0].chronic == pytest.approx(100.0)


# ------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"acute": 28, "chronic": 7},   # inverted
        {"acute": 7, "chronic": 7},    # equal
        {"acute": 0, "chronic": 28},   # zero
        {"acute": -1, "chronic": 28},  # negative
    ],
)
def test_bad_windows_rejected(kwargs):
    with pytest.raises(ValueError):
        rolling_acwr([100.0] * 30, **kwargs)


def test_negative_load_rejected():
    with pytest.raises(ValueError, match="negative load at index 2"):
        rolling_acwr([10.0, 10.0, -5.0])


def test_nan_load_rejected():
    with pytest.raises(ValueError, match="NaN load at index 1"):
        rolling_acwr([10.0, math.nan])


# ----------------------------------------------------------------- zones


@pytest.mark.parametrize(
    "ratio,zone",
    [
        (0.5, Zone.UNDERLOADED),
        (0.79, Zone.UNDERLOADED),
        (0.8, Zone.OPTIMAL),
        (1.3, Zone.OPTIMAL),
        (1.31, Zone.CAUTION),
        (1.5, Zone.CAUTION),
        (1.51, Zone.HIGH_RISK),
        (None, Zone.UNDEFINED),
        (math.nan, Zone.UNDEFINED),
    ],
)
def test_zone_boundaries(ratio, zone):
    assert classify(ratio) is zone


# ------------------------------------------------------- monotony / strain


def test_monotony_of_varied_week():
    # mean 150, sample sd sqrt(70000/6) = 108.01  ->  150 / 108.01
    assert monotony([100.0, 200.0, 50.0, 300.0, 0.0, 150.0, 250.0]) == pytest.approx(
        1.3888, abs=1e-3
    )


def test_monotony_undefined_for_identical_days():
    assert monotony([100.0] * 7) is None


def test_monotony_needs_two_days():
    assert monotony([100.0]) is None
    assert monotony([]) is None


def test_strain_is_total_load_times_monotony():
    loads = [100.0, 200.0, 50.0, 300.0, 0.0, 150.0, 250.0]
    assert strain(loads) == pytest.approx(sum(loads) * monotony(loads))


def test_strain_undefined_when_monotony_is():
    assert strain([100.0] * 7) is None
