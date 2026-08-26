"""Label bleed: the outcome, or something built from it, inside the features.

In workload modelling the classic form is the injury date sitting inside the
window the features are computed over. The seven days before an injury are
*different* — sessions get cut short, loads drop, an athlete is pulled at half
time — so a window that reaches the injury day encodes the injury in the load.
"""

from __future__ import annotations

import ast

from ruck.audit.astutils import (
    Module,
    assigned_names,
    call_name,
    literal,
    subscript_keys,
)
from ruck.audit.checks.base import (
    ModuleFacts,
    all_subscript_keys,
    find_window,
    is_label_name,
    is_probable_label_name,
    is_workload_name,
    normalise,
)
from ruck.audit.models import Category, Confidence, Finding, Severity

#: Feature names that are computed from the outcome by definition.
OUTCOME_DERIVED_FRAGMENTS: tuple[str, ...] = (
    "days_since_injury",
    "days_to_injury",
    "time_since_injury",
    "injury_count",
    "previous_injury",
    "prior_injury",
    "n_injuries",
    "injury_history",
    "return_to_play",
    "days_out",
    "games_missed",
    "sessions_missed",
    "availability",
)


def check(module: Module, facts: ModuleFacts) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Assign):
            findings.extend(_feature_matrix_holds_label(module, node, facts))
            findings.extend(_feature_from_label(module, node))
            findings.extend(_window_over_label(module, node))
            findings.extend(_outcome_derived_feature(module, node))
        if isinstance(node, ast.Compare):
            findings.extend(_inclusive_injury_filter(module, node))
    return findings


def _feature_matrix_holds_label(
    module: Module, node: ast.Assign, facts: ModuleFacts
) -> list[Finding]:
    """`X = df[[...]]` or `features = [...]` that still contains the outcome."""
    names = assigned_names(node)
    if not names:
        return []
    target = normalise(names[0])
    if target not in {
        "x",
        "x_train",
        "x_test",
        "features",
        "feature_cols",
        "feature_columns",
        "predictors",
    }:
        return []

    columns = subscript_keys(node.value)
    if not columns:
        parsed = literal(node.value)
        if isinstance(parsed, (list, tuple)):
            columns = [str(v) for v in parsed]
    leaked = sorted({c for c in columns if is_probable_label_name(c)})
    # `X = tracking[["x", "y"]]` is a coordinate pair, not a leaked label.
    if "y" in leaked and any(c.lower() == "x" for c in columns):
        leaked.remove("y")
    if not leaked:
        return []
    return [
        Finding(
            check_id="LB001",
            category=Category.LABEL_BLEED,
            title=f"Feature set contains the outcome column ({', '.join(leaked)})",
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                f"`{names[0]}` selects {', '.join(repr(c) for c in leaked)}, which is the "
                "thing being predicted. Any model fitted on this reports near-perfect "
                "skill and has learned nothing."
            ),
            fix=(
                "Build the feature list explicitly and assert the label is not in it, "
                "rather than dropping columns by name after the fact."
            ),
            evidence={"columns": leaked},
        )
    ]


#: Calls that legitimately take the label as an argument. `X_train, ..., y_train
#: = train_test_split(X, y)` is not a feature built from the outcome.
LABEL_CONSUMING_CALLS: frozenset[str] = frozenset(
    {
        "train_test_split",
        "fit",
        "fit_transform",
        "fit_resample",
        "fit_predict",
        "cross_val_score",
        "cross_validate",
        "resample",
        "score",
        "evaluate",
        "compute_class_weight",
        "to_categorical",
        "LabelEncoder",
        "value_counts",
        "unique",
        "nunique",
        "confusion_matrix",
        "classification_report",
        "roc_auc_score",
        "accuracy_score",
        "f1_score",
        "train",
        "split",
    }
)


def _feature_from_label(module: Module, node: ast.Assign) -> list[Finding]:
    """A non-label column whose value is computed from the label column."""
    names = assigned_names(node)
    if not names or any(is_label_name(name) for name in names):
        return []
    # A tuple target is almost always a splitter or an encoder returning both
    # halves; the label is an input to it, not an ingredient of a feature.
    if any(isinstance(target, (ast.Tuple, ast.List)) for target in node.targets):
        return []
    if _calls_label_consumer(node.value) or _is_row_filter(node.value):
        return []
    if _is_evaluation(names[0], node.value):
        return []
    referenced = sorted(
        {
            key
            for key in all_subscript_keys(node.value) + _names_in(node.value)
            if is_probable_label_name(key)
        }
    )
    if not referenced:
        return []
    if not is_workload_name(names[0]) and not _looks_like_feature(names[0]):
        return []
    return [
        Finding(
            check_id="LB002",
            category=Category.LABEL_BLEED,
            title=f"Feature `{names[0]}` is computed from the outcome",
            severity=Severity.CRITICAL,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                f"`{names[0]}` is derived from {', '.join(repr(r) for r in referenced)}. "
                "Even a heavily transformed function of the label is still the label; the "
                "model recovers it and the score is meaningless."
            ),
            fix=(
                "Compute the feature from workload and availability data recorded before "
                "the outcome window opens."
            ),
            evidence={"derived_from": referenced},
        )
    ]


def _window_over_label(module: Module, node: ast.Assign) -> list[Finding]:
    """A rolling window computed over a frame slice that includes the label."""
    window = find_window(node.value)
    if window is None:
        return []
    columns = all_subscript_keys(node.value)
    leaked = sorted({c for c in columns if is_label_name(c)})
    if not leaked:
        return []
    names = assigned_names(node)
    if names and is_label_name(names[0]):
        return []
    return [
        Finding(
            check_id="LB003",
            category=Category.LABEL_BLEED,
            title="Rolling window is computed over the outcome column",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                f"The window covers {', '.join(repr(c) for c in leaked)} alongside the "
                "workload columns, so the aggregate carries the outcome into a feature. "
                "A 28-day mean that includes an injury flag is a smoothed injury flag."
            ),
            fix="Select only workload columns before the window, by explicit list.",
            evidence={"columns": leaked, "window": window.size},
        )
    ]


def _outcome_derived_feature(module: Module, node: ast.Assign) -> list[Finding]:
    names = assigned_names(node)
    if not names:
        return []
    name = normalise(names[0])
    matched = [fragment for fragment in OUTCOME_DERIVED_FRAGMENTS if fragment in name]
    if not matched:
        return []
    return [
        Finding(
            check_id="LB005",
            category=Category.LABEL_BLEED,
            title=f"`{names[0]}` is defined by the outcome it is used to predict",
            severity=Severity.HIGH,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "Injury-history features are legitimate only when they are strictly "
                "backward-looking from the prediction point. Computed over the whole "
                "record — which is the default when the column is built once from the "
                "full frame — they encode injuries that have not happened yet. "
                "`days_since_injury` computed this way is negative before an injury and "
                "the model reads the sign."
            ),
            fix=(
                "Recompute the feature as of each prediction date, using only injuries "
                "recorded strictly before it, and unit-test the boundary."
            ),
            evidence={"fragment": matched[0]},
        )
    ]


def _inclusive_injury_filter(module: Module, node: ast.Compare) -> list[Finding]:
    """`df[df.date <= injury_date]` keeps the injury day's session in the window."""
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.LtE):
        return []
    left_names = _names_in(node.left) + subscript_keys(node.left)
    right_names = _names_in(node.comparators[0]) + subscript_keys(node.comparators[0])
    if not any(_is_date(name) for name in left_names):
        return []
    if not any(is_label_name(name) for name in right_names):
        return []
    return [
        Finding(
            check_id="LB004",
            category=Category.LABEL_BLEED,
            title="Feature window is inclusive of the injury date",
            severity=Severity.HIGH,
            confidence=Confidence.MEDIUM,
            file=module.relative,
            line=module.line(node),
            end_line=module.end_line(node),
            snippet=module.snippet(node),
            mechanism=(
                "`<=` keeps the injury day itself inside the feature window. That day's "
                "session is the one that was cut short, so its load is unusually low and "
                "unusually informative. The comparison should be strict."
            ),
            fix=(
                "Use `<` against the injury date, or better, end the window a full day "
                "before it so a mid-session injury cannot bleed in at all."
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


def _is_date(name: str) -> bool:
    text = normalise(name)
    return any(token in text for token in ("date", "day", "session", "timestamp"))


#: Name fragments that mean "this number describes the outcome" rather than
#: "this number is a feature": scores, rates and report tables.
EVALUATION_FRAGMENTS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "auc",
    "roc",
    "brier",
    "logloss",
    "log_loss",
    "f1",
    "mse",
    "rmse",
    "mae",
    "confusion",
    "report",
    "metric",
    "eval",
    "baseline",
    "base_rate",
    "positive_rate",
    "prevalence",
    "balance",
    "rate_index",
    "summary",
    "n_positive",
    "n_negative",
    "counts",
    "distribution",
)

#: Names that hold model output. A quantity computed from predictions and the
#: outcome together is a score, not a feature.
PREDICTION_FRAGMENTS: tuple[str, ...] = (
    "pred",
    "prediction",
    "predicted",
    "proba",
    "probability",
    "y_hat",
    "yhat",
    "score_out",
    "output",
    "fitted",
)


#: Calls that collapse rows. A number produced by one of these is a statistic
#: about the outcome, not a per-row feature that could leak into a model.
AGGREGATIONS: frozenset[str] = frozenset(
    {
        "mean",
        "sum",
        "count",
        "size",
        "std",
        "var",
        "min",
        "max",
        "median",
        "describe",
        "value_counts",
        "nunique",
        "quantile",
        "agg",
        "corr",
        "cov",
    }
)

#: Calls that keep one value per row, so an "aggregate" after them is still a
#: per-row feature.
ROW_ALIGNED: frozenset[str] = frozenset({"rolling", "ewm", "expanding", "transform", "shift"})


def _is_summary_of(value: ast.AST) -> bool:
    """True when an expression collapses rows rather than producing one per row."""
    calls = [call_name(inner) for inner in ast.walk(value) if isinstance(inner, ast.Call)]
    if any(name in ROW_ALIGNED for name in calls):
        return False
    return any(name in AGGREGATIONS for name in calls)


def _is_evaluation(name: str, value: ast.AST) -> bool:
    """True when an assignment is measuring the outcome rather than predicting it.

    A Brier score, a class-balance figure and a weekly injury-rate table are all
    computed from the label on purpose. Reporting them as label bleed is how a
    leakage checker teaches people to ignore it.
    """
    folded = normalise(name)
    if any(fragment in folded for fragment in EVALUATION_FRAGMENTS):
        return True
    if _is_summary_of(value):
        return True
    referenced = [normalise(n) for n in _names_in(value)] + [
        normalise(k) for k in all_subscript_keys(value)
    ]
    return any(
        any(fragment in candidate for fragment in PREDICTION_FRAGMENTS) for candidate in referenced
    )


def _is_row_filter(node: ast.AST) -> bool:
    """`panel[panel['date'] <= injury_date]` selects rows; it is not a feature.

    The inclusive-date version of this is a real finding, but LB004 reports it —
    reporting it twice under two names helps nobody.
    """
    return isinstance(node, ast.Subscript) and any(
        isinstance(inner, (ast.Compare, ast.BoolOp)) for inner in ast.walk(node.slice)
    )


def _calls_label_consumer(node: ast.AST) -> bool:
    """True when the expression hands the label to something that expects it."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            name = call_name(inner)
            if name in LABEL_CONSUMING_CALLS:
                return True
    return False


def _looks_like_feature(name: str) -> bool:
    """Names that read as an engineered feature rather than a working variable."""
    text = normalise(name)
    return any(
        token in text
        for token in (
            "feature",
            "ratio",
            "acwr",
            "index",
            "score",
            "risk",
            "monotony",
            "strain",
            "rolling",
            "mean",
            "sum",
            "acute",
            "chronic",
            "flag",
            "encoded",
        )
    )


def label_columns(facts: ModuleFacts) -> list[str]:
    return sorted(facts.label_names)
