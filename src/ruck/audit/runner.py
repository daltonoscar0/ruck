"""Walk a repository, run every check, merge with Hindsight if it is installed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ruck.audit import hindsight as hindsight_wrapper
from ruck.audit.astutils import Module, python_files
from ruck.audit.checks import CHECKS, collect_facts
from ruck.audit.checks.base import ModuleFacts
from ruck.audit.models import AuditResult, Finding


@dataclass
class RepoFacts:
    """Whole-repository facts, used to write the purge list."""

    workload_columns: set[str] = field(default_factory=set)
    label_columns: set[str] = field(default_factory=set)
    athlete_columns: set[str] = field(default_factory=set)
    time_columns: set[str] = field(default_factory=set)

    def absorb(self, facts: ModuleFacts) -> None:
        # Only names that were used to index a frame reach the purge list: it
        # tells a reader which columns to change, so a function name in it is
        # worse than useless.
        columns = facts.column_names
        self.workload_columns |= facts.workload_names & columns
        self.label_columns |= facts.label_names & columns
        self.athlete_columns |= facts.athlete_names & columns
        self.time_columns |= facts.time_names & columns


def audit_repo(
    root: str | Path,
    *,
    use_hindsight: bool = True,
    hindsight_executable: str | None = None,
    only: tuple[str, ...] | None = None,
) -> tuple[AuditResult, RepoFacts]:
    """Audit ``root`` read-only and return the findings and the repo's vocabulary.

    Args:
        use_hindsight: wrap the local Hindsight CLI when it is available.
        only: restrict to these check ids, e.g. ``("AC001", "LA002")``.
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    if root.is_file():
        files = [root]
        root = root.parent
    else:
        files = python_files(root)

    result = AuditResult(root=str(root))
    repo_facts = RepoFacts()

    for path in files:
        try:
            module = Module.parse(path, root)
        except SyntaxError as exc:
            result.errors.append(f"{path}: {exc.msg} (line {exc.lineno})")
            continue
        except OSError as exc:
            result.errors.append(f"{path}: {exc}")
            continue

        result.files_scanned += 1
        facts = collect_facts(module)
        repo_facts.absorb(facts)

        for name, check in CHECKS:
            if name not in result.checks_run:
                result.checks_run.append(name)
            try:
                result.findings.extend(check(module, facts))
            except Exception as exc:  # a broken check must not kill the audit
                result.errors.append(f"{module.relative}: check {name} failed: {exc}")

    if only:
        wanted = {c.upper() for c in only}
        result.findings = [f for f in result.findings if f.check_id.upper() in wanted]

    if use_hindsight:
        run = hindsight_wrapper.run(root, executable=hindsight_executable)
        result.hindsight_used = run.used
        result.hindsight_command = run.command
        if run.used:
            result.findings.extend(_deduplicate(result.findings, run.findings))
            result.checks_run.append("hindsight")
        elif run.available and run.error:
            result.errors.append(f"hindsight was found but not usable: {run.error}")

    return result, repo_facts


def _deduplicate(existing: list[Finding], incoming: list[Finding]) -> list[Finding]:
    """Drop incoming findings that sit on a line ruck already flagged.

    Both engines look for lookahead, so the same `shift(-1)` can come back
    twice. The native finding wins because its wording is written for a
    workload audience.
    """
    taken = {(f.file, f.line, f.category) for f in existing}
    out: list[Finding] = []
    for finding in incoming:
        if (finding.file, finding.line, finding.category) in taken:
            continue
        out.append(finding)
        taken.add((finding.file, finding.line, finding.category))
    return out
