"""Small helpers for reading pandas/sklearn code with the `ast` module.

The audit is static and strictly read-only: target files are parsed, never
imported and never executed. That matters when the target is a research repo
whose top-level code trains a model or hits a database on import.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Module:
    """One parsed Python file."""

    path: Path
    relative: str
    source: str
    tree: ast.Module

    @classmethod
    def parse(cls, path: Path, root: Path) -> Module:
        source = path.read_text(encoding="utf-8", errors="replace")
        return cls(
            path=path,
            relative=str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            source=source,
            tree=ast.parse(source, filename=str(path)),
        )

    def snippet(self, node: ast.AST) -> str:
        """The source text of ``node``, trimmed to one readable line."""
        text = ast.get_source_segment(self.source, node) or ""
        collapsed = " ".join(text.split())
        return collapsed if len(collapsed) <= 160 else collapsed[:157] + "..."

    def line(self, node: ast.AST) -> int:
        return int(getattr(node, "lineno", 0) or 0)

    def end_line(self, node: ast.AST) -> int | None:
        value = getattr(node, "end_lineno", None)
        return int(value) if value else None


def attr_path(node: ast.AST) -> str:
    """Dotted path for a Name/Attribute chain: ``pd.DataFrame.rolling``.

    Subscripts collapse to their base so ``df['load'].rolling`` reads as
    ``df.rolling``, which is what a check wants to match on.
    """
    parts: list[str] = []
    current: ast.AST | None = node
    while True:
        if isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        elif isinstance(current, ast.Name):
            parts.append(current.id)
            break
        elif isinstance(current, ast.Subscript):
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        else:
            break
    return ".".join(reversed(parts))


def call_name(node: ast.Call) -> str:
    """The final attribute or name being called: ``mean`` for ``x.rolling(3).mean()``."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def call_chain(node: ast.Call) -> list[str]:
    """Method names in a chained call, outermost last.

    ``df['load'].rolling(7).sum()`` gives ``["rolling", "sum"]``.
    """
    chain: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Call):
        chain.append(call_name(current))
        func = current.func
        current = func.value if isinstance(func, ast.Attribute) else func
        while isinstance(current, ast.Subscript):
            current = current.value
    return list(reversed(chain))


def keyword_value(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def literal(node: ast.AST | None) -> object | None:
    """The literal value of ``node``, or None if it is not a literal."""
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def first_arg(node: ast.Call, index: int = 0) -> ast.AST | None:
    return node.args[index] if len(node.args) > index else None


def int_arg(node: ast.Call, index: int = 0, keyword: str | None = None) -> int | None:
    """A positional or keyword integer argument, if it is a literal int."""
    value = literal(first_arg(node, index))
    if value is None and keyword is not None:
        value = literal(keyword_value(node, keyword))
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def subscript_keys(node: ast.AST) -> list[str]:
    """String keys used to subscript ``node``: ``df[['a','b']]`` gives ``['a','b']``."""
    keys: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Subscript):
        value = literal(current.slice)
        if isinstance(value, str):
            keys.append(value)
        elif isinstance(value, (list, tuple)):
            keys.extend(str(v) for v in value if isinstance(v, str))
        current = current.value
    return keys


def source_column(node: ast.AST) -> str | None:
    """The column a rolling expression is computed over, if it is written literally."""
    keys = subscript_keys(node)
    return keys[0] if keys else None


def base_name(node: ast.AST) -> str | None:
    """The root variable of an expression: ``df`` for ``df['x'].rolling(7)``."""
    current: ast.AST | None = node
    while True:
        if isinstance(current, ast.Name):
            return current.id
        if isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        else:
            return None


def assigned_names(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> list[str]:
    """Readable names for the assignment targets, including subscript keys."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Subscript):
            keys = subscript_keys(target)
            names.extend(keys or [base_name(target) or ""])
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                if isinstance(element, ast.Name):
                    names.append(element.id)
    return [name for name in names if name]


def iter_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def string_constants(tree: ast.AST) -> set[str]:
    """Every string literal in a subtree, for name-based heuristics."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
    return out


def enclosing_statement(tree: ast.Module, target: ast.AST) -> ast.stmt | None:
    """The statement a node belongs to, for reporting a whole line."""
    best: ast.stmt | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            line = getattr(target, "lineno", None)
            if start is None or end is None or line is None:
                continue
            if start <= line <= end and (
                best is None or (end - start) < (best.end_lineno - best.lineno)  # type: ignore[operator]
            ):
                best = node
    return best


def python_files(root: Path, *, skip: tuple[str, ...] = ()) -> list[Path]:
    """Every ``.py`` file under ``root``, minus the usual noise."""
    ignored = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        "site-packages",
        ".mypy_cache",
        ".tox",
        "node_modules",
    }
    ignored.update(skip)
    out: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        # Match on the path *below* the audit root only. Filtering on the whole
        # absolute path means a repository that happens to live under a
        # directory called `build` or `site-packages` audits to zero files -
        # which is exactly what happens to the bundled fixtures once ruck is
        # installed as a wheel.
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            parts = path.parts
        if any(part in ignored for part in parts):
            continue
        out.append(path)
    return out
