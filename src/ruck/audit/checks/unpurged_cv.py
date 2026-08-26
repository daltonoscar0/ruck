"""Cross-validation that lets one athlete-week inform another.

Workload data is a panel: many athletes, each a time series, with features
built from overlapping windows. Two things break a naive split. Shuffling puts
Tuesday in train and Wednesday in test, and a 28-day window means the two rows
are mostly the same data. And splitting without grouping puts the same athlete
on both sides, so the model can recognise the player rather than the pattern.
"""

from __future__ import annotations

import ast

from ruck.audit.astutils import Module, call_name, keyword_value, literal
from ruck.audit.checks.base import ModuleFacts
from ruck.audit.models import Category, Confidence, Finding, Severity

#: Splitters that ignore time order.
SHUFFLING_SPLITTERS: tuple[str, ...] = ("KFold", "StratifiedKFold", "RepeatedKFold", "ShuffleSplit")

#: Splitters that keep an athlete on one side of the split.
GROUP_AWARE: tuple[str, ...] = (
    "GroupKFold",
    "StratifiedGroupKFold",
    "GroupShuffleSplit",
    "LeaveOneGroupOut",
)


def check(module: Module, facts: ModuleFacts) -> list[Finding]:
    if not facts.has_model_fit:
        return []
    findings: list[Finding] = []
    group_aware_used = _uses_group_aware(module)

    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node)
        if name in SHUFFLING_SPLITTERS:
            findings.extend(_shuffling_splitter(module, node, name, facts))
        elif name == "train_test_split":
            findings.extend(_train_test_split(module, node, facts))
        elif name in {"cross_val_score", "cross_validate", "GridSearchCV", "RandomizedSearchCV"}:
            findings.extend(_default_cv(module, node, facts, group_aware_used))
        elif name == "TimeSeriesSplit":
            findings.extend(_no_embargo(module, node, facts))

    if facts.has_athletes and not group_aware_used:
        findings.extend(_missing_grouping(module, facts))
    return findings


def _uses_group_aware(module: Module) -> bool:
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Call) and call_name(node) in GROUP_AWARE:
            return True
        if isinstance(node, ast.Call) and keyword_value(node, "groups") is not None:
            return True
    return False


def _shuffling_splitter(
    module: Module, node: ast.Call, name: str, facts: ModuleFacts
) -> list[Finding]:
    shuffle = literal(keyword_value(node, "shuffle"))
    if shuffle is not True:
        return []
    return [
        Finding(
            check_id="CV001",
            category=Category.UNPURGED_CV,
            title=f"{name}(shuffle=True) on a time-ordered panel",
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH if facts.has_rolling else Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "Shuffling breaks the time order, so a fold can train on next week and "
                "test on this one."
                + (
                    " Worse here: rolling windows mean adjacent rows share most of their "
                    "input, so a test row's 28-day window overlaps a training row's by 27 "
                    "days. The fold boundary is not a boundary."
                    if facts.has_rolling
                    else ""
                )
            ),
            fix=(
                "Split forward in time (`TimeSeriesSplit`, or explicit date cut-offs), and "
                "purge a gap at least as long as the longest feature window so no test row "
                "shares data with a training row."
            ),
            evidence={"splitter": name},
        )
    ]


def _train_test_split(module: Module, node: ast.Call, facts: ModuleFacts) -> list[Finding]:
    shuffle = literal(keyword_value(node, "shuffle"))
    if shuffle is False:
        return []
    return [
        Finding(
            check_id="CV002",
            category=Category.UNPURGED_CV,
            title="train_test_split shuffles by default",
            severity=Severity.HIGH,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "`shuffle` defaults to True, so the test set is drawn uniformly from the "
                "whole season rather than from its end. The reported score is an average "
                "over a task the model will never face: it never gets to interpolate "
                "between weeks it has already seen."
            ),
            fix=(
                "Pass `shuffle=False` and split on a date, or use a grouped, time-ordered splitter."
            ),
            evidence={},
        )
    ]


def _default_cv(
    module: Module, node: ast.Call, facts: ModuleFacts, group_aware: bool
) -> list[Finding]:
    cv = keyword_value(node, "cv")
    literal_cv = literal(cv)
    if cv is not None and not isinstance(literal_cv, int):
        return []
    if group_aware:
        return []
    return [
        Finding(
            check_id="CV003",
            category=Category.UNPURGED_CV,
            title=f"{call_name(node)} uses the default K-fold split",
            severity=Severity.HIGH,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "With `cv` unset or an integer, scikit-learn builds a plain (stratified) "
                "K-fold. On an athlete-week panel that mixes athletes and weeks across the "
                "fold boundary in both directions."
            ),
            fix=(
                "Pass an explicit splitter that is both time-ordered and grouped by "
                "athlete, and pass `groups=` so it is honoured."
            ),
            evidence={"cv": literal_cv},
        )
    ]


def _no_embargo(module: Module, node: ast.Call, facts: ModuleFacts) -> list[Finding]:
    if not facts.has_rolling:
        return []
    gap_node = keyword_value(node, "gap")
    gap = literal(gap_node)
    if isinstance(gap, int) and gap > 0:
        return []
    if gap_node is not None and gap is None:
        # `gap=LONGEST_FEATURE_WINDOW` is an embargo whose size this checker
        # cannot read. Reporting "no gap" because it is not an integer literal
        # is simply false.
        return []
    return [
        Finding(
            check_id="CV005",
            category=Category.UNPURGED_CV,
            title="TimeSeriesSplit with no gap, but features use rolling windows",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "The split respects time order but leaves no embargo. The first test row "
                "sits one day after the last training row, and its 28-day chronic window "
                "contains 27 days that are in the training set. The fold boundary leaks "
                "for as many rows as the window is long."
            ),
            fix=(
                "Set `gap` to at least the longest feature window — 28 for a four-week "
                "chronic load, more if any feature looks further back."
            ),
            evidence={"gap": gap},
        )
    ]


def _missing_grouping(module: Module, facts: ModuleFacts) -> list[Finding]:
    """No group-aware splitter anywhere, in a file that clearly has athletes."""
    line = 0
    snippet = ""
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Call) and call_name(node) in {
            "cross_val_score",
            "cross_validate",
            "train_test_split",
            "GridSearchCV",
            "RandomizedSearchCV",
            "KFold",
            "StratifiedKFold",
            "TimeSeriesSplit",
        }:
            line = module.line(node)
            snippet = module.snippet(node)
            break
    if not line:
        return []
    return [
        Finding(
            check_id="CV004",
            category=Category.UNPURGED_CV,
            title="Athlete-level panel split without grouping by athlete",
            severity=Severity.MEDIUM,
            confidence=Confidence.LOW,
            file=module.relative,
            line=line,
            snippet=snippet,
            mechanism=(
                "The file works with athlete identifiers "
                f"({', '.join(sorted(facts.athlete_names)[:3])}) "
                "but no splitter is grouped by them and no `groups=` is passed. The same "
                "player appears in training and test, so part of what the model learns is "
                "that player's baseline. Squad-level scores are then optimistic for anyone "
                "new."
            ),
            fix=(
                "Use `StratifiedGroupKFold` or `GroupKFold` with `groups=df['player_id']`, "
                "or hold out whole athletes as well as later dates."
            ),
            evidence={"athlete_names": sorted(facts.athlete_names)[:5]},
        )
    ]
