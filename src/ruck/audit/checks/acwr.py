"""Acute:chronic workload ratio checks.

The rugby-specific section of the audit, and the one worth reading first.

An ACWR is an acute window divided by a chronic window over the same series.
In the usual formulation the acute window is *inside* the chronic one: the last
seven days appear in the numerator and again in the denominator. Writing the
chronic four-week block as a weekly mean, ``ACWR = A / ((A + B) / 4) = 4A/(A + B)``,
where ``B`` is the preceding three weeks. That is a bounded, monotone function
of ``A`` — it moves with acute load whether or not acute load has anything to
do with injury. Any model fitted on it inherits that mechanical relationship
and reports it as a finding.

The uncoupled form takes the chronic window from the period *before* the acute
one, so the two share no observations::

    acute   = load.rolling(7).sum()
    chronic = load.shift(7).rolling(21).sum()      # no overlap
    acwr    = acute / (chronic / 3)
"""

from __future__ import annotations

import ast

from ruck.audit.astutils import Module, assigned_names, keyword_value, literal
from ruck.audit.checks.base import (
    ModuleFacts,
    Window,
    find_pandas_windows,
    find_slice_windows,
    is_workload_name,
    mentions_acwr,
    normalise,
    resolve_window,
    scope_map,
)
from ruck.audit.models import Category, Confidence, Finding, Severity

#: The thresholds the "sweet spot" literature uses. Seeing them binned in code
#: is a strong signal the coupled ratio is being treated as a risk category.
SWEET_SPOT_VALUES: tuple[float, ...] = (0.8, 1.3, 1.5)

#: Guards that stop a near-zero chronic load producing an infinite ratio.
DENOMINATOR_GUARDS: tuple[str, ...] = (
    "clip",
    "replace",
    "where",
    "maximum",
    "fillna",
    "mask",
    "clip_lower",
    "nan_to_num",
)


def check(module: Module, facts: ModuleFacts) -> list[Finding]:
    findings: list[Finding] = []
    about_acwr = mentions_acwr(module)
    # Window variables are local, so the name -> window map is built per scope.
    scopes = scope_map(module.tree)
    window_maps: dict[int, dict[str, Window]] = {}

    def windows_for(node: ast.AST) -> dict[str, Window]:
        scope = scopes.get(id(node), module.tree)
        key = id(scope)
        if key not in window_maps:
            window_maps[key] = find_pandas_windows(scope, scopes)
        return window_maps[key]

    for node in ast.walk(module.tree):
        if isinstance(node, ast.Assign):
            findings.extend(_check_assignment(module, node, facts, windows_for(node)))
        elif isinstance(node, ast.Return) and node.value is not None:
            # `return acute / chronic` is as common as assigning it first.
            findings.extend(
                _check_ratio(module, node, node.value, facts, windows_for(node), label="the ratio")
            )
        if isinstance(node, ast.Call):
            findings.extend(_check_binning(module, node))
        if isinstance(node, ast.Compare):
            findings.extend(_check_threshold(module, node, about_acwr))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            findings.extend(_check_slice_windows(module, node, scopes))
            if about_acwr:
                findings.extend(_check_python_ewma(module, node))
    return _collapse_thresholds(findings)


def _collapse_thresholds(findings: list[Finding]) -> list[Finding]:
    """One AC005 per file, with a count.

    A `classify()` function has one comparison per band, and reporting each as
    its own finding turns one design decision into four rows.
    """
    kept: list[Finding] = []
    first: Finding | None = None
    extra = 0
    for finding in findings:
        if finding.check_id != "AC005":
            kept.append(finding)
        elif first is None:
            first = finding
            kept.append(finding)
        else:
            extra += 1
    if first is not None and extra:
        first.evidence["further_occurrences"] = extra
        first.title += f" ({extra + 1} thresholds in this file)"
    return kept


def _check_slice_windows(
    module: Module,
    scope: ast.FunctionDef | ast.AsyncFunctionDef,
    scopes: dict[int, ast.AST] | None = None,
) -> list[Finding]:
    """Coupled windows written as list slices rather than with pandas."""
    windows = find_slice_windows(scope, scopes)
    if len(windows) < 2:
        return []

    findings: list[Finding] = []
    seen: set[int] = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        numerator = _window_of(node.left, windows)
        denominator = _window_of(node.right, windows)
        if numerator is None or denominator is None:
            continue
        if numerator.sequence != denominator.sequence:
            continue
        if numerator.size_key == denominator.size_key:
            continue
        if numerator.end_offset != denominator.end_offset:
            continue
        line = module.line(node)
        if line in seen:
            continue
        seen.add(line)
        findings.append(
            Finding(
                check_id="AC002",
                category=Category.ACWR_COUPLING,
                title="Coupled acute:chronic ratio — both windows end on the same day",
                severity=Severity.CRITICAL,
                confidence=Confidence.HIGH,
                file=module.relative,
                line=line,
                end_line=module.end_line(node),
                snippet=module.snippet(node),
                mechanism=(
                    f"The two windows are slices of `{numerator.sequence}` that both end at "
                    f"the same index, with lengths `{numerator.size_key}` and "
                    f"`{denominator.size_key}`. The shorter window is therefore entirely "
                    "inside the longer one, and every day in the numerator is counted again "
                    "in the denominator. With the chronic block written as the acute window "
                    "plus the rest, the ratio reduces to a bounded increasing function of "
                    "acute load: it rises after a hard week as a matter of arithmetic, "
                    "before any athlete is involved."
                ),
                fix=_slice_fix(module, numerator, denominator),
                evidence={
                    "sequence": numerator.sequence,
                    "acute_window": numerator.size_key,
                    "chronic_window": denominator.size_key,
                    "style": "list-slice",
                },
            )
        )
    return findings


def _slice_fix(module: Module, acute, chronic) -> str:
    """A fix written in the same slice convention the source used.

    The uncoupled chronic window ends exactly where the acute one begins, so
    its upper bound is the acute slice's lower bound — whatever that happens to
    be spelled as. Assuming an `i - n + 1` convention and emitting it back is
    wrong for code written any other way, and a fix that is subtly off is worse
    than no fix at all.
    """
    acute_lower = module.snippet(acute.lower_node) if acute.lower_node is not None else None
    chronic_lower = module.snippet(chronic.lower_node) if chronic.lower_node is not None else None
    if acute_lower and chronic_lower:
        return (
            "End the chronic slice where the acute one begins, so the two share no "
            f"observations: `{chronic.sequence}[{chronic_lower} : {acute_lower}]`. "
            "Divide by its own length rather than the acute one's. Compute both the "
            "coupled and uncoupled ratios for one season and compare before choosing."
        )
    return (
        "End the chronic slice where the acute one begins rather than where it ends, "
        "so the two windows share no observations, and divide by its own length. "
        "Compute both the coupled and uncoupled ratios for one season and compare "
        "before choosing."
    )


def _window_of(node: ast.AST, windows: dict[str, object]):
    """The window an expression is computed over, following simple names."""
    if isinstance(node, ast.Name):
        return windows.get(node.id)
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in windows:
            return windows[inner.id]
    return None


def _check_python_ewma(
    module: Module, scope: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[Finding]:
    """Two hand-rolled EWMAs of one series, divided by each other."""
    smoothed: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = normalise(target.id)
            if not any(token in name for token in ("ewma", "ewm", "smooth", "exp_")):
                continue
            # A recurrence: the variable appears on both sides.
            if any(
                isinstance(inner, ast.Name) and inner.id == target.id
                for inner in ast.walk(node.value)
            ):
                smoothed.add(target.id)

    if len(smoothed) < 2:
        return []
    for node in ast.walk(scope):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        left = node.left.id if isinstance(node.left, ast.Name) else None
        right = node.right.id if isinstance(node.right, ast.Name) else None
        if left in smoothed and right in smoothed and left != right:
            return [
                Finding(
                    check_id="AC003",
                    category=Category.ACWR_COUPLING,
                    title="Exponentially weighted ACWR shares every observation between "
                    "numerator and denominator",
                    severity=Severity.HIGH,
                    confidence=Confidence.HIGH,
                    file=module.relative,
                    line=module.line(node),
                    end_line=module.end_line(node),
                    snippet=module.snippet(node),
                    mechanism=(
                        f"`{left}` and `{right}` are two exponentially weighted means of the "
                        "same load series with different smoothing constants, so every "
                        "session contributes to both. Removing the hard window edge removes "
                        "the discontinuity, not the coupling: the ratio is still a function "
                        "of the series divided by a smoother function of the same series."
                    ),
                    fix=(
                        "Seed the chronic EWMA from load up to the start of the acute "
                        "period, or report acute and chronic load as two features and let "
                        "the model combine them."
                    ),
                    evidence={"acute": left, "chronic": right, "style": "hand-rolled EWMA"},
                )
            ]
    return []


def _check_assignment(
    module: Module,
    node: ast.Assign,
    facts: ModuleFacts,
    known_windows: dict[str, Window] | None = None,
) -> list[Finding]:
    names = assigned_names(node)
    return _check_ratio(
        module, node, node.value, facts, known_windows, label=names[0] if names else "acwr"
    )


def _check_ratio(
    module: Module,
    node: ast.stmt,
    expression: ast.AST,
    facts: ModuleFacts,
    known_windows: dict[str, Window] | None = None,
    *,
    label: str,
) -> list[Finding]:
    """Check one ``acute / chronic`` expression, however it is being used."""
    names = [label]
    ratio = _as_ratio(expression)
    if ratio is None:
        return []
    known_windows = known_windows or {}
    numerator, denominator = ratio
    acute = resolve_window(numerator, known_windows)
    chronic = resolve_window(denominator, known_windows)
    if acute is None or chronic is None:
        return []

    same_series = _same_series(acute, chronic, facts)
    if not same_series:
        return []

    findings: list[Finding] = []
    label = names[0] if names else "acwr"

    if acute.kind == "ewm" and chronic.kind == "ewm":
        findings.append(
            Finding(
                check_id="AC003",
                category=Category.ACWR_COUPLING,
                title="Exponentially weighted ACWR shares every observation between "
                "numerator and denominator",
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                file=module.relative,
                line=module.line(node),
                end_line=module.end_line(node),
                snippet=module.snippet(node),
                mechanism=(
                    "Both the acute and the chronic term are exponentially weighted means "
                    "of the same series, so every session contributes to both. The ratio "
                    "is a smoothed function of the series divided by a more smoothed "
                    "function of the same series; it moves with recent load by "
                    "construction. EWMA removes the hard window edge, not the coupling."
                ),
                fix=(
                    "Compute the chronic term from load up to the start of the acute "
                    "period: `chronic = load.shift(acute_span).ewm(span=chronic_span).mean()`. "
                    "Better, drop the ratio and give the model acute and chronic load as "
                    "two features, letting it find its own combination."
                ),
                evidence={
                    "variable": label,
                    "acute_span": acute.size,
                    "chronic_span": chronic.size,
                },
            )
        )
        return findings

    overlap = _overlap_days(acute, chronic)
    if overlap is None or overlap > 0:
        contained = overlap is not None and acute.size is not None and overlap >= acute.size
        findings.append(
            Finding(
                check_id="AC001",
                category=Category.ACWR_COUPLING,
                title=(
                    "Coupled acute:chronic ratio — the acute window is inside the chronic window"
                    if contained
                    else "Coupled acute:chronic ratio — the windows share observations"
                ),
                severity=Severity.CRITICAL,
                confidence=Confidence.HIGH if overlap else Confidence.MEDIUM,
                file=module.relative,
                line=module.line(node),
                end_line=module.end_line(node),
                snippet=module.snippet(node),
                mechanism=_coupling_mechanism(acute, chronic, overlap),
                fix=(
                    "Take the chronic window from the period before the acute one so the "
                    f"two share no sessions: `chronic = load.shift({acute.size or 7})"
                    f".rolling({_uncoupled_chronic(acute, chronic)}).sum()`, then divide by "
                    "the number of weeks it spans. Report the uncoupled ratio next to the "
                    "coupled one; if the effect only exists in the coupled version, it was "
                    "the algebra."
                ),
                evidence={
                    "variable": label,
                    "acute_window": acute.size,
                    "chronic_window": chronic.size,
                    "shared_observations": overlap,
                },
            )
        )

    if not (_has_guard(node) or _has_guard(acute.node) or _has_guard(chronic.node)):
        findings.append(
            Finding(
                check_id="AC004",
                category=Category.ACWR_COUPLING,
                title="ACWR divides by chronic load with no floor on the denominator",
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                file=module.relative,
                line=module.line(node),
                end_line=module.end_line(node),
                snippet=module.snippet(node),
                mechanism=(
                    "Chronic load is near zero for a returning player, pre-season, or the "
                    "first four weeks of any athlete's record. Dividing by it produces "
                    "ratios in the hundreds or an infinity, and those rows then dominate "
                    "whatever is fitted on them. The extreme values are an artefact of the "
                    "denominator, not a training spike."
                ),
                fix=(
                    "Require a minimum chronic period before emitting a ratio "
                    "(`rolling(28, min_periods=28)`), and drop rather than clip the rows "
                    "that do not have one. Clipping keeps a fabricated value in the sample."
                ),
                evidence={"variable": label},
            )
        )

    if acute.covers_current_row and facts.has_label:
        findings.append(
            Finding(
                check_id="AC006",
                category=Category.LABEL_BLEED,
                title="Acute window includes the day the outcome is recorded",
                severity=Severity.HIGH,
                confidence=Confidence.MEDIUM,
                file=module.relative,
                line=module.line(node),
                end_line=module.end_line(node),
                snippet=module.snippet(node),
                mechanism=(
                    "The acute window ends on the current row, so a session's own load is a "
                    "feature for that session's outcome. When an injury occurs during a "
                    "session, the truncated or elevated load of that session is in the "
                    "window — the feature partly encodes the label. Models trained this way "
                    "look predictive and cannot be used prospectively, because on the "
                    "morning of a session that load does not exist yet."
                ),
                fix=(
                    "Shift the whole feature block by one observation before the ratio: "
                    "`load.shift(1).rolling(7).sum()`. Every feature should be computable "
                    "the morning before the session it describes."
                ),
                evidence={"variable": label, "acute_window": acute.size},
            )
        )
    return findings


def _as_ratio(node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """Unwrap ``a / b``, seeing through a trailing scale like ``/ 4``."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return None
    denominator = node.right
    # `acute / (chronic / 4)` — the weekly-average form.
    if (
        isinstance(denominator, ast.BinOp)
        and isinstance(denominator.op, ast.Div)
        and isinstance(literal(denominator.right), (int, float))
    ):
        denominator = denominator.left
    return node.left, denominator


def _same_series(acute: Window, chronic: Window, facts: ModuleFacts) -> bool:
    """True when both windows are computed over the same workload series."""
    if acute.column and chronic.column:
        return normalise(acute.column) == normalise(chronic.column)
    if acute.frame and chronic.frame and acute.frame == chronic.frame:
        return True
    # Two windows over unnamed series in a file full of workload columns is
    # still worth flagging, but only at lower confidence.
    return bool(facts.workload_names) and acute.frame == chronic.frame


def _overlap_days(acute: Window, chronic: Window) -> int | None:
    """How many observations the two windows share, if both sizes are literal.

    Windows are half-open intervals ending at ``-shift``: an unshifted
    ``rolling(7)`` covers the seven rows up to and including the current one.
    """
    if acute.size is None or chronic.size is None:
        return None
    acute_end = -acute.shifted_by
    acute_start = acute_end - acute.size + 1
    chronic_end = -chronic.shifted_by
    chronic_start = chronic_end - chronic.size + 1
    return max(0, min(acute_end, chronic_end) - max(acute_start, chronic_start) + 1)


def _uncoupled_chronic(acute: Window, chronic: Window) -> int:
    if acute.size and chronic.size and chronic.size > acute.size:
        return chronic.size - acute.size
    return 21


def _coupling_mechanism(acute: Window, chronic: Window, overlap: int | None) -> str:
    if acute.size and chronic.size and overlap:
        if overlap < acute.size:
            return (
                f"The acute window covers {acute.size} observations and the chronic window "
                f"{chronic.size}, and {overlap} of them fall in both. The chronic term is "
                "offset but not far enough to clear the acute one, so part of the numerator "
                "is still counted in the denominator and the ratio moves with acute load "
                "for arithmetic reasons as well as physiological ones. Offsetting the "
                f"chronic window by the full {acute.size} removes the overlap entirely."
            )
        weeks = chronic.size / acute.size if acute.size else 4
        return (
            f"The acute window is the last {acute.size} observations and the chronic window "
            f"is the last {chronic.size}, so {overlap} observations are counted in both the "
            "numerator and the denominator. Writing the chronic block as acute plus the "
            f"rest, the ratio reduces to {weeks:.0f}A / (A + B): a bounded, increasing "
            "function of acute load whatever B does. It rises after a hard week because of "
            "the algebra, not because of the athlete. An association between this quantity "
            "and injury is therefore not evidence of one between load and injury."
        )
    return (
        "The acute and chronic windows are computed over the same series and the chronic "
        "one is not offset past the acute one, so observations are shared between the "
        "numerator and the denominator and the ratio moves with acute load by construction."
    )


def _has_guard(node: ast.Assign) -> bool:
    """True when the statement protects against a near-zero denominator."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            name = (
                inner.func.attr
                if isinstance(inner.func, ast.Attribute)
                else (inner.func.id if isinstance(inner.func, ast.Name) else "")
            )
            if name in DENOMINATOR_GUARDS:
                return True
            if name == "rolling" and literal(keyword_value(inner, "min_periods")) is not None:
                return True
        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add):
            value = literal(inner.right)
            if isinstance(value, float) and 0 < value < 1e-2:
                return True
    return False


def _check_binning(module: Module, node: ast.Call) -> list[Finding]:
    """`pd.cut(acwr, bins=[0, 0.8, 1.3, ...])` — the sweet-spot categories."""
    name = (
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else (node.func.id if isinstance(node.func, ast.Name) else "")
    )
    if name not in {"cut", "qcut", "digitize"}:
        return []
    values: list[float] = []
    for candidate in list(node.args) + [k.value for k in node.keywords]:
        parsed = literal(candidate)
        if isinstance(parsed, (list, tuple)):
            values.extend(float(v) for v in parsed if isinstance(v, (int, float)))
    if not any(abs(v - target) < 1e-9 for v in values for target in SWEET_SPOT_VALUES):
        return []
    return [
        Finding(
            check_id="AC005",
            category=Category.ACWR_COUPLING,
            title="ACWR binned into sweet-spot categories",
            severity=Severity.MEDIUM,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "Cutting the ratio at 0.8 / 1.3 / 1.5 turns a mechanically coupled "
                "quantity into a risk category, which hides the coupling behind a label. "
                "The bin edges are inherited from published work whose ratios were built "
                "the same coupled way, so agreement with it is not corroboration."
            ),
            fix=(
                "Keep the ratio continuous, report the uncoupled version alongside it, and "
                "let any thresholds be fitted on this squad rather than imported."
            ),
            evidence={"bin_edges": values},
        )
    ]


def _check_threshold(module: Module, node: ast.Compare, about_acwr: bool) -> list[Finding]:
    """A bare `acwr > 1.5` comparison, which is the same idea without `pd.cut`."""
    left = node.left
    name = getattr(left, "id", None) or getattr(left, "attr", None)
    if name is None:
        keys = [
            c.value
            for c in ast.walk(left)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        ]
        name = keys[0] if keys else None
    if name is None:
        return []
    named_acwr = "acwr" in normalise(name)
    # In a file that is plainly about acute:chronic load, a threshold on a
    # variable called `ratio` is the same finding under a different name.
    generic = normalise(name) in {"ratio", "value", "acwr_value", "acute_chronic"}
    if not named_acwr and not (about_acwr and generic):
        return []
    values = [literal(comparator) for comparator in node.comparators]
    hits = [
        v
        for v in values
        if isinstance(v, (int, float)) and any(abs(v - t) < 1e-9 for t in SWEET_SPOT_VALUES)
    ]
    if not hits:
        return []
    return [
        Finding(
            check_id="AC005",
            category=Category.ACWR_COUPLING,
            title="ACWR compared against an imported sweet-spot threshold",
            severity=Severity.LOW,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "The threshold comes from studies whose ratio was coupled the same way "
                "this one is. Applying it here reproduces the artefact rather than "
                "testing it."
            ),
            fix="Fit any threshold on this squad's own uncoupled ratio, or drop it.",
            evidence={"threshold": hits[0], "variable": str(name)},
        )
    ]


def workload_series_names(facts: ModuleFacts) -> list[str]:
    """Workload-looking column names, for the purge list."""
    return sorted(name for name in facts.workload_names if is_workload_name(name))
