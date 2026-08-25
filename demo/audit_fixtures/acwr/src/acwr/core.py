"""Acute:chronic workload ratio and related training-load metrics.

Implements the two ratio methods in common use:

* **Rolling average (RA)** — Hulin et al. (2014). Acute load is the mean of the
  last ``acute`` days; chronic load the mean of the last ``chronic`` days.
* **Exponentially weighted moving average (EWMA)** — Williams et al. (2017),
  which weights recent sessions more heavily and does not discard the day that
  falls out of the window.

Loads are unitless: session-RPE × minutes, TRIMP, WHOOP strain, distance —
anything monotonic in "how hard was that day". Rest days are ``0.0``, and they
matter, so do not omit them.

Pure standard library. No numpy, no pandas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

__all__ = [
    "Zone",
    "AcwrPoint",
    "rolling_acwr",
    "ewma_acwr",
    "monotony",
    "strain",
    "classify",
]


class Zone(str, Enum):
    """Interpretation bands for an ACWR value.

    The boundaries follow the commonly cited "sweet spot" of 0.8–1.3. They are a
    population-level heuristic, not a diagnosis — see the README.
    """

    UNDERLOADED = "underloaded"
    OPTIMAL = "optimal"
    CAUTION = "caution"
    HIGH_RISK = "high_risk"
    UNDEFINED = "undefined"


@dataclass(frozen=True)
class AcwrPoint:
    """One day of computed load."""

    day: int
    load: float
    acute: float
    chronic: float
    ratio: float | None
    zone: Zone

    @property
    def is_ready(self) -> bool:
        """False while the chronic window is still filling up."""
        return self.ratio is not None


def classify(ratio: float | None) -> Zone:
    """Map a ratio onto its interpretation band."""
    if ratio is None or math.isnan(ratio):
        return Zone.UNDEFINED
    if ratio < 0.8:
        return Zone.UNDERLOADED
    if ratio <= 1.3:
        return Zone.OPTIMAL
    if ratio <= 1.5:
        return Zone.CAUTION
    return Zone.HIGH_RISK


def _validate(loads: Sequence[float], acute: int, chronic: int) -> None:
    if acute < 1 or chronic < 1:
        raise ValueError("window lengths must be >= 1")
    if acute >= chronic:
        raise ValueError(
            f"acute window ({acute}) must be shorter than chronic ({chronic})"
        )
    for i, value in enumerate(loads):
        if value < 0:
            raise ValueError(f"negative load at index {i}: {value!r}")
        if math.isnan(value):
            raise ValueError(f"NaN load at index {i}")


def rolling_acwr(
    loads: Iterable[float],
    acute: int = 7,
    chronic: int = 28,
    *,
    require_full_window: bool = True,
) -> list[AcwrPoint]:
    """Rolling-average ACWR, one point per day.

    Args:
        loads: daily training loads, oldest first. Include rest days as ``0.0``.
        acute: length of the acute (fatigue) window in days.
        chronic: length of the chronic (fitness) window in days.
        require_full_window: when True (the default) the ratio is ``None`` until
            ``chronic`` days of history exist. Setting it False will happily
            divide by a partial window, which reads high and is the single most
            common way people mislead themselves with this metric.

    Returns:
        A list of :class:`AcwrPoint`, same length as ``loads``.
    """
    series = [float(x) for x in loads]
    _validate(series, acute, chronic)

    points: list[AcwrPoint] = []
    for i in range(len(series)):
        acute_window = series[max(0, i - acute + 1) : i + 1]
        chronic_window = series[max(0, i - chronic + 1) : i + 1]

        acute_mean = sum(acute_window) / len(acute_window)
        chronic_mean = sum(chronic_window) / len(chronic_window)

        ratio: float | None
        if require_full_window and (i + 1) < chronic:
            ratio = None
        elif chronic_mean == 0:
            # No chronic load at all: the ratio is genuinely undefined rather
            # than infinite. Returning None keeps callers from plotting inf.
            ratio = None
        else:
            ratio = acute_mean / chronic_mean

        points.append(
            AcwrPoint(
                day=i,
                load=series[i],
                acute=acute_mean,
                chronic=chronic_mean,
                ratio=ratio,
                zone=classify(ratio),
            )
        )
    return points


def ewma_acwr(
    loads: Iterable[float],
    acute: int = 7,
    chronic: int = 28,
    *,
    require_full_window: bool = True,
) -> list[AcwrPoint]:
    """Exponentially weighted ACWR (Williams et al., 2017).

    Uses the standard smoothing constant ``lambda = 2 / (N + 1)`` for each
    window. Both EWMAs are seeded with the first day's load.

    Args and returns mirror :func:`rolling_acwr`.
    """
    series = [float(x) for x in loads]
    _validate(series, acute, chronic)

    if not series:
        return []

    lambda_a = 2.0 / (acute + 1)
    lambda_c = 2.0 / (chronic + 1)

    ewma_a = series[0]
    ewma_c = series[0]

    points: list[AcwrPoint] = []
    for i, load in enumerate(series):
        if i > 0:
            ewma_a = load * lambda_a + ewma_a * (1 - lambda_a)
            ewma_c = load * lambda_c + ewma_c * (1 - lambda_c)

        ratio: float | None
        if require_full_window and (i + 1) < chronic:
            ratio = None
        elif ewma_c == 0:
            ratio = None
        else:
            ratio = ewma_a / ewma_c

        points.append(
            AcwrPoint(
                day=i,
                load=load,
                acute=ewma_a,
                chronic=ewma_c,
                ratio=ratio,
                zone=classify(ratio),
            )
        )
    return points


def monotony(loads: Sequence[float]) -> float | None:
    """Foster's training monotony: mean daily load / standard deviation.

    High monotony (>2.0) means every day looks the same, which is associated
    with worse adaptation even when total load is reasonable.

    Returns ``None`` if fewer than two days are supplied, or if every day is
    identical (zero variance makes the metric undefined rather than infinite).
    """
    values = [float(x) for x in loads]
    if len(values) < 2:
        return None

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    sd = math.sqrt(variance)
    if sd == 0:
        return None
    return mean / sd


def strain(loads: Sequence[float]) -> float | None:
    """Foster's training strain: total load × monotony.

    Returns ``None`` when monotony is undefined.
    """
    m = monotony(loads)
    if m is None:
        return None
    return sum(float(x) for x in loads) * m
