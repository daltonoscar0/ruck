"""Survivorship: the squad the model was fitted on is not the squad it will see.

The characteristic workload-model version is dropping players once they are
injured. What remains is the population that stayed fit, and the model learns
what fit players' weeks look like — which is not the same thing as learning
what precedes an injury.
"""

from __future__ import annotations

import ast

from ruck.audit.astutils import Module, call_name, keyword_value, literal
from ruck.audit.checks.base import (
    ModuleFacts,
    is_label_name,
    normalise,
)
from ruck.audit.models import Category, Confidence, Finding, Severity

#: Column-name fragments that describe how a season ended, not how it was going.
END_OF_PERIOD_FRAGMENTS: tuple[str, ...] = (
    "games_played",
    "matches_played",
    "appearances",
    "minutes_played",
    "sessions_completed",
    "completed_season",
    "full_season",
    "available_all",
    "total_minutes",
    "season_total",
)

#: Status values that mean "still standing at the end".
SURVIVOR_VALUES: tuple[str, ...] = ("active", "available", "fit", "selected", "completed")


def check(module: Module, facts: ModuleFacts) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Compare):
            findings.extend(_outcome_filter(module, node))
            findings.extend(_end_of_period_filter(module, node))
        if isinstance(node, ast.Call):
            findings.extend(_wholesale_dropna(module, node, facts))
    return findings


def _outcome_filter(module: Module, node: ast.Compare) -> list[Finding]:
    """`df[df.injured == 0]` and friends: selecting the population by the outcome."""
    if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
        return []
    names = _names_in(node.left) + _strings_in(node.left)
    if not any(is_label_name(name) for name in names):
        return []
    value = literal(node.comparators[0])
    negative = value in (0, False, "0", "no", "none", None)
    if isinstance(node.ops[0], ast.NotEq):
        negative = not negative
    if not negative and not isinstance(value, str):
        return []
    return [
        Finding(
            check_id="SV001",
            category=Category.SURVIVORSHIP,
            title="Population is filtered by the outcome being predicted",
            severity=Severity.CRITICAL,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "The rows kept are chosen by the injury column, so the sample is "
                "conditioned on the outcome. If uninjured rows are kept for feature "
                "building, the features describe uninjured weeks and the contrast the "
                "model needs has been removed; if injured players are dropped entirely, "
                "the cohort is the players who survived the season."
            ),
            fix=(
                "Keep every athlete-week in the panel and let the label mark which ones "
                "ended in an injury. If athlete-weeks must be excluded — a player on "
                "loan, a mid-season signing — exclude them on a rule that does not "
                "mention the outcome, and record the count."
            ),
            evidence={"comparison": module.snippet(node)},
        )
    ]


def _end_of_period_filter(module: Module, node: ast.Compare) -> list[Finding]:
    """`df[df.games_played > 10]`: a cohort chosen with end-of-season knowledge."""
    names = _names_in(node.left) + _strings_in(node.left)
    matched = [
        name
        for name in names
        if any(fragment in normalise(name) for fragment in END_OF_PERIOD_FRAGMENTS)
    ]
    if not matched:
        values = [literal(c) for c in node.comparators]
        strings = [v for v in values if isinstance(v, str) and normalise(v) in SURVIVOR_VALUES]
        if not strings:
            return []
        matched = [str(strings[0])]
    return [
        Finding(
            check_id="SV003",
            category=Category.SURVIVORSHIP,
            title=f"Cohort selected on a season total (`{matched[0]}`)",
            severity=Severity.HIGH,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                f"`{matched[0]}` is only known once the season is over, and it is lower "
                "for exactly the players who got injured. Requiring a minimum therefore "
                "removes the injured players from the sample before the model sees them. "
                "In September, the squad this model will be applied to includes everyone."
            ),
            fix=(
                "Select the cohort on information available at the start of the window "
                "(contracted, in the squad, has n weeks of prior data) rather than on "
                "totals accumulated across it."
            ),
            evidence={"column": matched[0]},
        )
    ]


def _wholesale_dropna(module: Module, node: ast.Call, facts: ModuleFacts) -> list[Finding]:
    """`df.dropna()` with no subset, in a file that has athletes and a label."""
    if call_name(node) != "dropna":
        return []
    if node.args or keyword_value(node, "subset") is not None:
        return []
    if not (facts.has_athletes and facts.has_label):
        return []
    return [
        Finding(
            check_id="SV002",
            category=Category.SURVIVORSHIP,
            title="Unrestricted dropna() removes athlete-weeks non-randomly",
            severity=Severity.MEDIUM,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "Missing GPS is not missing at random. It is concentrated in the weeks "
                "around injury, illness and rehab — the weeks the model most needs. "
                "Dropping every row with any null quietly removes them, and also removes "
                "the first 27 days of every athlete's record, where the 28-day chronic "
                "window has not filled."
            ),
            fix=(
                "Drop on an explicit subset, count what was dropped and report it, and "
                "check whether the dropped rows are disproportionately close to injuries."
            ),
            evidence={},
        )
    ]


def _names_in(node: ast.AST) -> list[str]:
    out: list[str] = []
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name):
            out.append(inner.id)
        elif isinstance(inner, ast.Attribute):
            out.append(inner.attr)
    return out


def _strings_in(node: ast.AST) -> list[str]:
    return [
        inner.value
        for inner in ast.walk(node)
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
    ]
