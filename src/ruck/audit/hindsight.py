"""Wrapper for a local Hindsight installation.

Hindsight audits trading-research repositories for the same family of problems
this tool looks for in workload models, so when it is on PATH ruck defers to it
and adds only the checks it does not have — chiefly the ACWR coupling section.

The spec for this tool calls the CLI as ``hindsight run <repo>``. The installed
CLI's subcommand is ``hindsight audit <repo>``. Rather than pick one and break
on the other, the wrapper probes ``audit`` first, then ``run``, and falls back
to the native checks if neither works. That is recorded in the report so a
reader knows which engine produced which finding.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ruck.audit.models import Finding

#: Subcommands to try, in order. `audit` is what the CLI implements today.
SUBCOMMANDS: tuple[str, ...] = ("audit", "run")

TIMEOUT_SECONDS = 300


@dataclass
class HindsightRun:
    """What happened when we tried to shell out to hindsight."""

    available: bool = False
    used: bool = False
    executable: str | None = None
    command: str | None = None
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    error: str | None = None


def find_cli(explicit: str | None = None) -> str | None:
    """Locate the hindsight executable, if there is one."""
    if explicit:
        return explicit if Path(explicit).exists() or shutil.which(explicit) else None
    return shutil.which("hindsight")


def run(
    target: Path, *, executable: str | None = None, timeout: int = TIMEOUT_SECONDS
) -> HindsightRun:
    """Run hindsight over ``target`` and adapt its findings.

    Never raises: a missing CLI, an unknown subcommand or a crash all come back
    as a :class:`HindsightRun` with ``used=False`` and an explanation, and the
    caller carries on with the native checks.
    """
    binary = find_cli(executable)
    if binary is None:
        return HindsightRun(error="hindsight is not on PATH")

    result = HindsightRun(available=True, executable=binary)
    attempts: list[str] = []
    for subcommand in SUBCOMMANDS:
        command = [binary, subcommand, str(target), "--format", "json"]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            attempts.append(f"{subcommand}: {exc}")
            continue

        # Exit code 1 means "findings were found", which is a success for us.
        if completed.returncode not in (0, 1):
            attempts.append(
                f"{subcommand}: exit {completed.returncode} {completed.stderr.strip()[:160]}"
            )
            continue
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            attempts.append(f"{subcommand}: output was not JSON ({exc})")
            continue

        result.used = True
        result.command = " ".join(command)
        result.findings = [Finding.from_hindsight(item) for item in payload.get("findings", [])]
        scanned = payload.get("files_scanned")
        result.files_scanned = int(scanned) if isinstance(scanned, int) else 0
        return result

    result.error = "; ".join(attempts) or "hindsight produced no usable output"
    return result
