"""Shared vocabulary and window analysis for the native checks."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from ruck.audit.astutils import (
    Module,
    attr_path,
    base_name,
    call_chain,
    int_arg,
    keyword_value,
    literal,
    source_column,
    subscript_keys,
)

#: Name fragments that mean "this is the thing we are trying to predict".
LABEL_TOKENS: tuple[str, ...] = (
    "injur",
    "injured",
    "is_injured",
    "time_loss",
    "timeloss",
    "missed",
    "unavailable",
    "soft_tissue",
    "hamstring",
    "acl",
    "concussion",
    "label",
    "target",
    "outcome",
    "days_to_injury",
    "days_until",
    "will_injure",
)

#: Name fragments that mean "this is a GPS or session workload measure".
WORKLOAD_TOKENS: tuple[str, ...] = (
    "load",
    "workload",
    "distance",
    "hsr",
    "high_speed",
    "highspeed",
    "sprint",
    "accel",
    "decel",
    "player_load",
    "playerload",
    "rpe",
    "srpe",
    "duration",
    "minutes",
    "gps",
    "impact",
    "collision",
    "metres",
    "meters",
    "exertion",
    "trimp",
    "session",
)

#: Name fragments that identify an athlete, and therefore a grouping unit.
ATHLETE_TOKENS: tuple[str, ...] = (
    "player",
    "athlete",
    "subject",
    "participant",
    "squad_number",
    "person",
)

#: Name fragments that mean a point in time.
TIME_TOKENS: tuple[str, ...] = ("date", "day", "time", "week", "session_no", "timestamp")

#: Methods whose string arguments are column names.
COLUMN_TAKING_CALLS: frozenset[str] = frozenset(
    {
        "groupby",
        "sort_values",
        "set_index",
        "reset_index",
        "drop",
        "dropna",
        "drop_duplicates",
        "merge",
        "join",
        "pivot",
        "pivot_table",
        "melt",
        "value_counts",
        "nunique",
        "agg",
        "rename",
        "astype",
        "read_csv",
        "parse_dates",
        "filter",
    }
)

_WORD = re.compile(r"[^a-z0-9]+")


def normalise(name: object) -> str:
    return _WORD.sub("_", str(name).lower())


def looks_like(name: object, tokens: tuple[str, ...]) -> bool:
    """True when a folded name contains one of ``tokens``.

    Short tokens have to sit on an underscore boundary. Substring matching
    `tri` against `metrics` and `stride`, or `acl` against `dataclass`, fills
    the purge list with function names.
    """
    text = normalise(name)
    for token in tokens:
        if len(token) >= 5:
            if token in text:
                return True
        elif re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", text):
            return True
    return False


_Y_VARIABLE = re.compile(r"^y(_|$)")

#: Names that contain a label token but are plotting calls, not outcomes.
NOT_LABELS: frozenset[str] = frozenset(
    {
        "xlabel",
        "ylabel",
        "clabel",
        "zlabel",
        "set_xlabel",
        "set_ylabel",
        "labelsize",
        "labelpad",
        "labelcolor",
        "ticklabel",
        "labelbottom",
        "labelleft",
    }
)


def is_label_name(name: object) -> bool:
    text = normalise(name)
    if text in NOT_LABELS:
        return False
    # `y`, `y_train`, `y_enc`, `y_resampled`: the conventional name for the
    # thing being predicted, in every shape people write it.
    if _Y_VARIABLE.match(text):
        return True
    return looks_like(name, LABEL_TOKENS)


def is_workload_name(name: object) -> bool:
    return looks_like(name, WORKLOAD_TOKENS)


def is_athlete_name(name: object) -> bool:
    """True for a name that identifies *who*, not *how much*.

    `player_load` is a GPS metric, not a grouping key, and a purge list that
    tells you to `groupby("player_load")` has stopped being useful.
    """
    return looks_like(name, ATHLETE_TOKENS) and not looks_like(name, WORKLOAD_TOKENS)


def is_time_name(name: object) -> bool:
    return looks_like(name, TIME_TOKENS)


@dataclass
class ModuleFacts:
    """Cheap whole-file facts the checks condition on.

    Several checks are only meaningful in a file that has both a label and a
    time axis; asking "is this CV scheme purged?" of a plotting script produces
    noise. Collecting the facts once keeps each check honest about its
    preconditions.
    """

    label_names: set[str]
    workload_names: set[str]
    athlete_names: set[str]
    time_names: set[str]
    has_rolling: bool
    has_model_fit: bool
    #: Names that were actually used to index a frame or read as an attribute,
    #: as opposed to any identifier that happened to match a token. The purge
    #: list is written from these, because it names columns to a reader who
    #: will go looking for them.
    column_names: set[str] = field(default_factory=set)

    @property
    def has_label(self) -> bool:
        return bool(self.label_names)

    @property
    def has_athletes(self) -> bool:
        return bool(self.athlete_names)


#: A column name is short, has no line breaks, and reads like a name rather
#: than a sentence. Without this, every docstring that mentions "load" ends up
#: in the purge list as a GPS column.
_COLUMN_LIKE = re.compile(r"^[A-Za-z][A-Za-z0-9_\- ]{0,39}$")


def is_column_like(value: object) -> bool:
    """True when a string literal could plausibly be a dataframe column."""
    text = str(value)
    if text != text.strip() or "\n" in text:
        return False
    # A dot means a filename or a dotted path, not a column.
    if "." in text:
        return False
    if not _COLUMN_LIKE.fullmatch(text):
        return False
    return len(text.split()) <= 4


def collect_facts(module: Module) -> ModuleFacts:
    """Scan a module once for the names and shapes the checks need."""
    labels: set[str] = set()
    workloads: set[str] = set()
    athletes: set[str] = set()
    times: set[str] = set()
    has_rolling = False
    has_fit = False

    # Strings used to index something are columns; strings used anywhere else
    # are prose until proven otherwise.
    indexed: set[str] = set()
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Subscript):
            indexed.update(subscript_keys(node))
        elif isinstance(node, ast.Attribute):
            indexed.add(node.attr)
        elif isinstance(node, ast.Call) and _called(node) in COLUMN_TAKING_CALLS:
            # `panel.sort_values(["player_id", "date"])` names columns just as
            # much as `panel["player_id"]` does.
            for argument in list(node.args) + [k.value for k in node.keywords]:
                value = literal(argument)
                if isinstance(value, str):
                    indexed.add(value)
                elif isinstance(value, (list, tuple)):
                    indexed.update(v for v in value if isinstance(v, str))

    for node in ast.walk(module.tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value
            if name not in indexed and not is_column_like(name):
                continue
            if is_label_name(name):
                labels.add(name)
            if is_workload_name(name):
                workloads.add(name)
            if is_athlete_name(name):
                athletes.add(name)
            if is_time_name(name):
                times.add(name)
        elif isinstance(node, ast.Name):
            if is_label_name(node.id):
                labels.add(node.id)
            if is_workload_name(node.id):
                workloads.add(node.id)
        elif isinstance(node, ast.Attribute):
            if is_label_name(node.attr):
                labels.add(node.attr)
            if is_athlete_name(node.attr):
                athletes.add(node.attr)
            if is_time_name(node.attr):
                times.add(node.attr)
        elif isinstance(node, ast.Call):
            name = _called(node)
            if name in {"rolling", "ewm", "expanding"}:
                has_rolling = True
            if name in {
                "fit",
                "fit_transform",
                "cross_val_score",
                "cross_validate",
                "fit_predict",
                "train",
            }:
                has_fit = True

    return ModuleFacts(
        label_names=labels,
        workload_names=workloads,
        athlete_names=athletes,
        time_names=times,
        has_rolling=has_rolling,
        has_model_fit=has_fit,
        column_names={name for name in indexed if is_column_like(name)},
    )


def _called(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


@dataclass
class Window:
    """A rolling / ewm window found in an expression."""

    node: ast.Call
    kind: str  # "rolling" | "ewm" | "expanding"
    size: int | None
    halflife: float | None
    column: str | None
    frame: str | None
    centered: bool
    shifted_by: int
    chain: list[str]

    @property
    def covers_current_row(self) -> bool:
        """True when the window includes the observation it is a feature for."""
        return self.shifted_by <= 0


def find_window(node: ast.AST) -> Window | None:
    """Describe the rolling window in ``node``, if there is one.

    Walks *down* the expression, so ``df['load'].shift(1).rolling(7).sum()``
    reports window 7 shifted by 1, and the checks can tell an honest
    feature from one that reads today's value.
    """
    call = _outermost_window_call(node)
    if call is None:
        return None

    kind = _called(call)
    size = int_arg(call, 0, "window") if kind == "rolling" else int_arg(call, 0, "span")
    halflife = literal(keyword_value(call, "halflife"))
    centered = bool(literal(keyword_value(call, "center")))

    shifted = 0
    inner: ast.AST | None = call.func.value if isinstance(call.func, ast.Attribute) else None
    while isinstance(inner, ast.Call):
        name = _called(inner)
        if name == "shift":
            amount = int_arg(inner, 0, "periods")
            shifted += amount if amount is not None else 1
        inner = inner.func.value if isinstance(inner.func, ast.Attribute) else None

    return Window(
        node=call,
        kind=kind,
        size=size,
        halflife=float(halflife) if isinstance(halflife, (int, float)) else None,
        column=source_column(call),
        frame=base_name(call),
        centered=centered,
        shifted_by=shifted,
        chain=call_chain(call) if isinstance(node, ast.Call) else [kind],
    )


def find_pandas_windows(scope: ast.AST) -> dict[str, Window]:
    """Map variable names to the rolling window they were assigned from.

    `acute = load.rolling(7).sum()` followed by `acwr = acute / chronic` is the
    common shape, and a checker that only looks at one statement at a time sees
    neither window at the division.
    """
    windows: dict[str, Window] = {}
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        name = target.id if isinstance(target, ast.Name) else None
        if name is None:
            continue
        window = find_window(node.value)
        if window is not None:
            windows[name] = window
    return windows


def resolve_window(node: ast.AST, known: dict[str, Window]) -> Window | None:
    """The window an expression denotes, following one hop through a variable."""
    window = find_window(node)
    if window is not None:
        return window
    if isinstance(node, ast.Name):
        return known.get(node.id)
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in known:
            return known[inner.id]
    return None


def all_subscript_keys(node: ast.AST) -> list[str]:
    """Every literal string key subscripted anywhere inside an expression."""
    keys: list[str] = []
    for inner in ast.walk(node):
        if isinstance(inner, ast.Subscript):
            keys.extend(subscript_keys(inner))
    return keys


def _outermost_window_call(node: ast.AST) -> ast.Call | None:
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call) and _called(inner) in {"rolling", "ewm", "expanding"}:
            return inner
    return None


@dataclass
class SliceWindow:
    """A trailing window written as a list slice rather than with pandas.

    Plenty of load code is pure standard library::

        acute_window   = series[max(0, i - acute + 1) : i + 1]
        chronic_window = series[max(0, i - chronic + 1) : i + 1]

    Both slices end at the same index, so the acute window is contained in the
    chronic one — the same coupling ``.rolling()`` produces, and invisible to a
    checker that only knows pandas.
    """

    node: ast.Subscript
    sequence: str
    size_expr: str | None
    size_value: int | None
    end_offset: int
    index_var: str | None

    @property
    def size_key(self) -> str:
        return self.size_expr or (str(self.size_value) if self.size_value is not None else "?")


def find_slice_windows(scope: ast.AST) -> dict[str, SliceWindow]:
    """Map variable names to the trailing slice window they are assigned from.

    Also follows one level of aggregation, so ``acute = sum(w) / len(w)`` is
    recorded as the same window as ``w``.
    """
    direct: dict[str, SliceWindow] = {}
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        window = parse_slice_window(node.value)
        if window is not None:
            direct[target.id] = window

    resolved = dict(direct)
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id in resolved:
            continue
        source = _aggregated_from(node.value, resolved)
        if source is not None:
            resolved[target.id] = source
    return resolved


#: Aggregations that turn a window into a single number without changing which
#: observations it covers.
_AGGREGATORS = frozenset({"sum", "mean", "average", "max", "min", "median", "len", "nanmean"})


def _aggregated_from(node: ast.AST, known: dict[str, SliceWindow]) -> SliceWindow | None:
    """The window an aggregate expression is computed over, if there is one."""
    found: SliceWindow | None = None
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            name = _called(inner)
            if name not in _AGGREGATORS:
                continue
            for argument in inner.args:
                if isinstance(argument, ast.Name) and argument.id in known:
                    found = known[argument.id]
                else:
                    window = parse_slice_window(argument)
                    if window is not None:
                        found = window
    return found


def parse_slice_window(node: ast.AST) -> SliceWindow | None:
    """Read ``series[max(0, i - N + 1) : i + 1]`` as a trailing window."""
    if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
        return None
    sequence = node.value.id if isinstance(node.value, ast.Name) else None
    if sequence is None:
        return None
    lower = node.slice.lower
    upper = node.slice.upper
    if lower is None or upper is None:
        return None

    low_index, subtracted, _ = _read_bound(lower)
    high_index, high_subtracted, high_const = _read_bound(upper)
    if low_index is None or high_index is None or low_index != high_index:
        return None
    if subtracted is None or high_subtracted is not None:
        return None

    size_value = subtracted if isinstance(subtracted, int) else None
    size_expr = subtracted if isinstance(subtracted, str) else None
    return SliceWindow(
        node=node,
        sequence=sequence,
        size_expr=size_expr,
        size_value=size_value,
        # Slices are half-open, so `: i + 1` ends on the current index.
        end_offset=high_const - 1,
        index_var=low_index,
    )


def _read_bound(node: ast.AST) -> tuple[str | None, str | int | None, int]:
    """Decompose a slice bound into (index variable, subtracted window, constant)."""
    index: str | None = None
    subtracted: str | int | None = None
    constant = 0

    def walk(current: ast.AST, sign: int) -> None:
        nonlocal index, subtracted, constant
        if isinstance(current, ast.Call) and _called(current) in {"max", "min"}:
            for argument in current.args:
                if isinstance(argument, ast.Constant):
                    continue
                walk(argument, sign)
            return
        if isinstance(current, ast.BinOp):
            if isinstance(current.op, ast.Add):
                walk(current.left, sign)
                walk(current.right, sign)
                return
            if isinstance(current.op, ast.Sub):
                walk(current.left, sign)
                walk(current.right, -sign)
                return
            return
        if isinstance(current, ast.Name):
            if sign > 0 and index is None:
                index = current.id
            elif sign < 0:
                subtracted = current.id
            return
        if isinstance(current, ast.Constant) and isinstance(current.value, int):
            if sign < 0 and subtracted is None:
                subtracted = current.value
            else:
                constant += sign * current.value

    walk(node, 1)
    return index, subtracted, constant


def mentions_acwr(module: Module) -> bool:
    """True when a file is about acute:chronic load at all."""
    haystack = module.source.lower()
    return any(token in haystack for token in ("acwr", "acute", "chronic"))


def frame_columns(node: ast.AST) -> list[str]:
    """Column names a subscript expression selects."""
    return subscript_keys(node)


def dotted(node: ast.AST) -> str:
    return attr_path(node)
