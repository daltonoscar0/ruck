"""Lookahead: features that could not have been computed at the time.

The test every one of these applies is the same. Stand on the morning of a
session, with only what the squad knew then. Could this number have been
produced? If it needs tomorrow's GPS file, next week's availability sheet, or
the whole season's distribution, it is lookahead, and the model's accuracy in
backtest is not accuracy it will have on Monday.
"""

from __future__ import annotations

import ast

from ruck.audit.astutils import Module, assigned_names, call_name, int_arg, keyword_value, literal
from ruck.audit.checks.base import (
    ModuleFacts,
    find_window,
    is_label_name,
    is_probable_label_name,
)
from ruck.audit.models import Category, Confidence, Finding, Severity

#: sklearn transformers whose `fit` learns statistics from the data it sees.
FITTED_TRANSFORMERS: tuple[str, ...] = (
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "Normalizer",
    "PowerTransformer",
    "QuantileTransformer",
    "KBinsDiscretizer",
    "SimpleImputer",
    "IterativeImputer",
    "KNNImputer",
    "PCA",
    "SelectKBest",
    "TargetEncoder",
)


def check(module: Module, facts: ModuleFacts) -> list[Finding]:
    findings: list[Finding] = []
    label_lines = _label_assignment_lines(module)
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Call):
            findings.extend(_negative_shift(module, node, label_lines))
            findings.extend(_centered_window(module, node))
            findings.extend(_backfill(module, node))
            findings.extend(_interpolate(module, node))
            findings.extend(_negative_diff(module, node))
        if isinstance(node, ast.Assign):
            findings.extend(_window_covers_today(module, node, facts))
    findings.extend(_fit_before_split(module))
    return findings


def _derivation(module: Module) -> dict[str, set[str]]:
    """For each variable, the variables it was (transitively) computed from.

    Enough to tell `scaler.fit(X); X_scaled = scaler.fit_transform(X);
    train_test_split(X_scaled, ...)` — where the fit really does see the test
    rows — from `scaler.fit(X_train)`, where it does not.
    """
    direct: dict[str, set[str]] = {}
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Assign):
            continue
        sources = {inner.id for inner in ast.walk(node.value) if isinstance(inner, ast.Name)}
        for name in assigned_names(node):
            direct.setdefault(name, set()).update(sources)

    resolved: dict[str, set[str]] = {}

    def expand(name: str, seen: frozenset[str]) -> set[str]:
        if name in resolved:
            return resolved[name]
        if name in seen:
            return set()
        out = set(direct.get(name, set()))
        for parent in list(out):
            out |= expand(parent, seen | {name})
        if name not in seen:
            resolved[name] = out
        return out

    return {name: expand(name, frozenset()) for name in direct}


def _label_assignment_lines(module: Module) -> set[int]:
    """Lines whose assignment target is the outcome being constructed.

    `panel["injury_next_7d"] = panel.groupby(...)["injured"].shift(-7)` is the
    textbook way to build a forward label, and this check's own fix text tells
    people to write exactly that. Flagging it as critical lookahead trains
    readers to ignore the check.
    """
    lines: set[int] = set()
    for node in ast.walk(module.tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = assigned_names(node)
            if names and any(is_probable_label_name(name) for name in names):
                start = module.line(node)
                end = module.end_line(node) or start
                lines.update(range(start, end + 1))
    return lines


def _negative_shift(module: Module, node: ast.Call, label_lines: set[int]) -> list[Finding]:
    if call_name(node) != "shift":
        return []
    periods = int_arg(node, 0, "periods")
    if periods is None or periods >= 0:
        return []
    if module.line(node) in label_lines:
        return []
    return [
        Finding(
            check_id="LA001",
            category=Category.LOOKAHEAD,
            title=f"Negative shift pulls {abs(periods)} future observation(s) backwards",
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                f"`shift({periods})` moves values from {abs(periods)} row(s) in the future "
                "onto the current row. If the result is a feature, the model is reading "
                "sessions that have not happened. If it is a label, that is correct and "
                "intended — but then nothing built from the same shifted frame may be a "
                "feature."
            ),
            fix=(
                "Use a negative shift only to construct the label, in its own statement, "
                "and build features from positive shifts of the unshifted frame."
            ),
            evidence={"periods": periods},
        )
    ]


def _centered_window(module: Module, node: ast.Call) -> list[Finding]:
    if call_name(node) not in {"rolling", "ewm"}:
        return []
    if not bool(literal(keyword_value(node, "center"))):
        return []
    size = int_arg(node, 0, "window")
    half = f"{size // 2} " if size else ""
    return [
        Finding(
            check_id="LA002",
            category=Category.LOOKAHEAD,
            title="Centred rolling window averages across the current observation",
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                f"`center=True` places the window symmetrically, so roughly {half}"
                "observations after the current row are inside it. A centred seven-day "
                "load average on Tuesday contains Thursday and Friday. Smoothing this way "
                "is fine for a chart and never fine for a feature."
            ),
            fix="Drop `center=True`; a trailing window is the causal one.",
            evidence={"window": size},
        )
    ]


def _backfill(module: Module, node: ast.Call) -> list[Finding]:
    name = call_name(node)
    method = literal(keyword_value(node, "method"))
    is_bfill = name in {"bfill", "backfill"} or (
        name == "fillna" and isinstance(method, str) and method in {"bfill", "backfill"}
    )
    if not is_bfill:
        return []
    return [
        Finding(
            check_id="LA003",
            category=Category.LOOKAHEAD,
            title="Backward fill copies later values onto earlier rows",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "Backfilling a gap in a workload series fills it with the next recorded "
                "session. Missing GPS is not missing at random — it clusters around "
                "injuries, illness and rest — so the filled value is drawn from exactly "
                "the period the model is trying to predict."
            ),
            fix=(
                "Forward-fill (`ffill`) if carrying the last known value is defensible, or "
                "leave the gap and let the model see a missing indicator. Say in the "
                "write-up which one you chose."
            ),
            evidence={"call": name},
        )
    ]


def _interpolate(module: Module, node: ast.Call) -> list[Finding]:
    if call_name(node) != "interpolate":
        return []
    direction = literal(keyword_value(node, "limit_direction"))
    if isinstance(direction, str) and direction == "forward":
        return []
    return [
        Finding(
            check_id="LA004",
            category=Category.LOOKAHEAD,
            title="Interpolation fills gaps from both sides",
            severity=Severity.MEDIUM,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "pandas interpolates between the values on either side of a gap, so a "
                "filled Wednesday is partly made of Friday. Over a missed week around an "
                "injury, that is the outcome leaking into the features."
            ),
            fix="Pass `limit_direction='forward'`, or leave the gap and flag it.",
            evidence={"limit_direction": direction},
        )
    ]


def _negative_diff(module: Module, node: ast.Call) -> list[Finding]:
    if call_name(node) != "diff":
        return []
    periods = int_arg(node, 0, "periods")
    if periods is None or periods >= 0:
        return []
    return [
        Finding(
            check_id="LA005",
            category=Category.LOOKAHEAD,
            title=f"`diff({periods})` differences against a future observation",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "A negative period differences the current row against a later one, so the "
                "result encodes what happens next."
            ),
            fix="Use a positive period to difference against the past.",
            evidence={"periods": periods},
        )
    ]


def _window_covers_today(module: Module, node: ast.Assign, facts: ModuleFacts) -> list[Finding]:
    """A rolling feature whose window ends on the row it describes."""
    if not facts.has_label:
        return []
    names = assigned_names(node)
    if not names or any(is_label_name(name) for name in names):
        return []
    window = find_window(node.value)
    if window is None or window.kind == "expanding" or not window.covers_current_row:
        return []
    if window.size is None or window.centered:
        # A centred window is a worse problem and LA002 already reports it.
        return []
    # Only flag things that are actually stored as features: a column write, or
    # a variable whose name reads like one. An intermediate local that feeds a
    # ratio is reported where the ratio is.
    writes_column = any(isinstance(target, ast.Subscript) for target in node.targets)
    if not writes_column and "_" not in names[0]:
        return []
    return [
        Finding(
            check_id="LA007",
            category=Category.LOOKAHEAD,
            title="Rolling feature window ends on the observation it is a feature for",
            severity=Severity.MEDIUM,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                f"`{names[0]}` is a {window.size}-observation window with no shift, so the "
                "current session's own load is inside it. For a same-day outcome that is a "
                "feature containing part of the label, and prospectively the value does not "
                "exist until the session is over."
            ),
            fix=(
                f"Shift before the window: `.shift(1).rolling({window.size})`. If the "
                "outcome is genuinely next-day, say so explicitly and keep the shift "
                "anyway, so the feature is computable in the morning."
            ),
            evidence={"variable": names[0], "window": window.size},
        )
    ]


def _fit_before_split(module: Module) -> list[Finding]:
    """A transformer fitted on the data that is split afterwards.

    Textual order is not execution order — two helper functions can appear in
    either order and behave identically — so the test is not "which line comes
    first". It is whether the thing being fitted is the same variable that is
    later handed to the splitter. Fitting `X_train` is correct however high up
    the file it sits; fitting `X` and then splitting `X` is the bug.
    """
    split_line: int | None = None
    split_inputs: set[str] = set()
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Call) and call_name(node) in {
            "train_test_split",
            "split",
            "TimeSeriesSplit",
            "KFold",
        }:
            line = module.line(node)
            split_line = line if split_line is None else min(split_line, line)
            for argument in node.args:
                if isinstance(argument, ast.Name):
                    split_inputs.add(argument.id)
    if split_line is None or not split_inputs:
        return []

    # The frame handed to the splitter is usually the transformer's *output*,
    # not the frame it was fitted on, so follow the assignments backwards.
    derives_from = _derivation(module)
    tainted = set(split_inputs)
    for name in split_inputs:
        tainted |= derives_from.get(name, set())

    constructed: dict[str, str] = {}
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            transformer = call_name(node.value)
            if transformer in FITTED_TRANSFORMERS:
                for name in assigned_names(node):
                    constructed[name] = transformer

    findings: list[Finding] = []
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Call) or call_name(node) not in {"fit", "fit_transform"}:
            continue
        line = module.line(node)
        fitted = [a.id for a in node.args if isinstance(a, ast.Name)]
        if not any(name in tainted for name in fitted):
            continue
        owner = node.func.value if isinstance(node.func, ast.Attribute) else None
        owner_name = getattr(owner, "id", None)
        transformer = constructed.get(owner_name or "")
        if transformer is None and not isinstance(node.func, ast.Attribute):
            continue
        if transformer is None:
            continue
        findings.append(
            Finding(
                check_id="LA006",
                category=Category.PREPROCESSING,
                title=f"{transformer} fitted on the full dataset before the split",
                severity=Severity.HIGH,
                confidence=Confidence.MEDIUM,
                file=module.relative,
                line=line,
                end_line=module.end_line(node),
                snippet=module.snippet(node),
                mechanism=(
                    f"`{owner_name}.{call_name(node)}` is fitted on "
                    f"`{', '.join(n for n in fitted if n in tainted)}`, which is the "
                    f"same data handed to the splitter at line {split_line}. The "
                    "transformer's statistics — means, quantiles, imputation values, "
                    "selected features — are therefore computed over rows that end up in "
                    "the test set, so the test set is no longer held out. On a workload "
                    "panel this is usually worth a few points of AUC that do not survive "
                    "deployment."
                ),
                fix=(
                    "Split first, then fit on the training rows only, or wrap the "
                    "transformer and the estimator in a `Pipeline` and fit the pipeline "
                    "inside the cross-validation loop."
                ),
                evidence={
                    "transformer": transformer,
                    "split_line": split_line,
                    "fitted_on": sorted(set(fitted) & tainted),
                },
            )
        )
    return findings
