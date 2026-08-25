"""The native checks.

Each module exposes ``check(module, facts) -> list[Finding]`` and is pure: it
parses, it never imports or executes the target.
"""

from __future__ import annotations

from collections.abc import Callable

from ruck.audit.astutils import Module
from ruck.audit.checks import acwr, label_bleed, lookahead, survivorship, unpurged_cv
from ruck.audit.checks.base import ModuleFacts, collect_facts
from ruck.audit.models import Finding

CheckFn = Callable[[Module, ModuleFacts], list[Finding]]

#: Run order is report order within a file: the rugby-specific check first,
#: because it is the one the reader came for.
CHECKS: tuple[tuple[str, CheckFn], ...] = (
    ("acwr", acwr.check),
    ("lookahead", lookahead.check),
    ("label_bleed", label_bleed.check),
    ("survivorship", survivorship.check),
    ("unpurged_cv", unpurged_cv.check),
)

CHECK_IDS: tuple[str, ...] = (
    "AC001",
    "AC003",
    "AC004",
    "AC005",
    "AC006",
    "LA001",
    "LA002",
    "LA003",
    "LA004",
    "LA005",
    "LA006",
    "LA007",
    "LB001",
    "LB002",
    "LB003",
    "LB004",
    "LB005",
    "SV001",
    "SV002",
    "SV003",
    "CV001",
    "CV002",
    "CV003",
    "CV004",
    "CV005",
)

__all__ = ["CHECKS", "CHECK_IDS", "CheckFn", "ModuleFacts", "collect_facts"]
