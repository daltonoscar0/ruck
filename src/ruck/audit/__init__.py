"""`ruck audit`: read a workload/injury-model repo and report what leaks."""

from __future__ import annotations

from pathlib import Path

from ruck.audit.hindsight import HindsightRun
from ruck.audit.hindsight import find_cli as find_hindsight
from ruck.audit.models import (
    AuditResult,
    Category,
    Confidence,
    Finding,
    Severity,
    Source,
)
from ruck.audit.report import render
from ruck.audit.runner import RepoFacts, audit_repo

__all__ = [
    "AuditResult",
    "Category",
    "Confidence",
    "Finding",
    "HindsightRun",
    "RepoFacts",
    "Severity",
    "Source",
    "audit",
    "audit_repo",
    "find_hindsight",
    "render",
]


def audit(
    target: str | Path,
    *,
    out: str | Path | None = None,
    use_hindsight: bool = True,
    hindsight_executable: str | None = None,
    only: tuple[str, ...] | None = None,
    generated_at: str | None = None,
) -> tuple[AuditResult, str]:
    """Audit ``target`` and return the result and its markdown report.

    Writes the markdown to ``out`` when given.
    """
    result, facts = audit_repo(
        target,
        use_hindsight=use_hindsight,
        hindsight_executable=hindsight_executable,
        only=only,
    )
    markdown = render(result, facts, generated_at=generated_at)
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return result, markdown
