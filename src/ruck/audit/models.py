"""The finding model shared by the native checks and the Hindsight wrapper.

Field names deliberately mirror `hindsight`'s JSON output so a wrapped run
merges into a native run without translation. Every finding carries a
``file`` and a ``line``: a leakage report with no citation is an opinion, and
the report renderer refuses to print one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(StrEnum):
    LOOKAHEAD = "lookahead"
    LABEL_BLEED = "label-bleed"
    SURVIVORSHIP = "survivorship"
    UNPURGED_CV = "unpurged-cv"
    ACWR_COUPLING = "acwr-coupling"
    PREPROCESSING = "preprocessing-leak"
    OTHER = "other"


class Source(StrEnum):
    NATIVE = "ruck"
    HINDSIGHT = "hindsight"


@dataclass
class Finding:
    """One problem, at one place, with a reason and a fix."""

    check_id: str
    category: Category
    title: str
    severity: Severity
    confidence: Confidence
    file: str
    line: int
    snippet: str = ""
    mechanism: str = ""
    fix: str = ""
    source: Source = Source.NATIVE
    end_line: int | None = None
    evidence: dict[str, object] = field(default_factory=dict)

    def citation(self) -> str:
        return f"{self.file}:{self.line}"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("category", "severity", "confidence", "source"):
            data[key] = getattr(self, key).value
        return data

    @classmethod
    def from_hindsight(cls, payload: dict[str, object]) -> Finding:
        """Adapt one entry of `hindsight audit --format json`."""
        try:
            severity = Severity(str(payload.get("severity", "medium")))
        except ValueError:
            severity = Severity.MEDIUM
        try:
            confidence = Confidence(str(payload.get("confidence", "medium")))
        except ValueError:
            confidence = Confidence.MEDIUM
        line = payload.get("line")
        return cls(
            check_id=str(payload.get("check_id", "HS???")),
            category=_map_category(str(payload.get("category", ""))),
            title=str(payload.get("title", "hindsight finding")),
            severity=severity,
            confidence=confidence,
            file=str(payload.get("file", "")),
            line=int(line) if isinstance(line, (int, float)) else 0,
            snippet=str(payload.get("snippet", "") or ""),
            mechanism=str(payload.get("mechanism", "") or ""),
            fix=str(payload.get("fix", "") or ""),
            source=Source.HINDSIGHT,
            end_line=payload.get("end_line") if isinstance(payload.get("end_line"), int) else None,
            evidence=dict(payload.get("evidence") or {})
            if isinstance(payload.get("evidence"), dict)
            else {},
        )


#: Hindsight's taxonomy is written for trading research. These are the pairs
#: that mean the same thing in a workload context; everything else lands in
#: `other` with the original label kept in the title.
_CATEGORY_MAP: dict[str, Category] = {
    "lookahead": Category.LOOKAHEAD,
    "look-ahead": Category.LOOKAHEAD,
    "label-leak": Category.LABEL_BLEED,
    "label-leakage": Category.LABEL_BLEED,
    "target-leak": Category.LABEL_BLEED,
    "survivorship": Category.SURVIVORSHIP,
    "survivor-bias": Category.SURVIVORSHIP,
    "cv-leak": Category.UNPURGED_CV,
    "cross-validation": Category.UNPURGED_CV,
    "unpurged-cv": Category.UNPURGED_CV,
    "preprocessing-leak": Category.PREPROCESSING,
}


def _map_category(raw: str) -> Category:
    return _CATEGORY_MAP.get(raw.strip().lower(), Category.OTHER)


@dataclass
class AuditResult:
    """A whole audit run."""

    root: str
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    checks_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    hindsight_used: bool = False
    hindsight_command: str | None = None

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (f.severity.rank, f.file, f.line, f.check_id))

    def counts_by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
        return counts

    def counts_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.category.value] = counts.get(finding.category.value, 0) + 1
        return counts

    def uncited(self) -> list[Finding]:
        """Findings without a usable ``file:line``; the report drops these."""
        return [f for f in self.findings if not f.file or not f.line]
