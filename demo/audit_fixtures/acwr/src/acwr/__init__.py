"""acwr — acute:chronic workload ratio, without the spreadsheet.

    >>> from acwr import rolling_acwr
    >>> points = rolling_acwr([300] * 21 + [900] * 7)
    >>> round(points[-1].ratio, 2)
    2.0
    >>> points[-1].zone.value
    'high_risk'
"""

from .core import (
    AcwrPoint,
    Zone,
    classify,
    ewma_acwr,
    monotony,
    rolling_acwr,
    strain,
)

__version__ = "0.1.0"

__all__ = [
    "AcwrPoint",
    "Zone",
    "classify",
    "ewma_acwr",
    "monotony",
    "rolling_acwr",
    "strain",
    "__version__",
]
