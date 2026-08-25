"""Vendor the public injury-model repositories `ruck audit demo/` runs against.

Clones each repository at its current head, records the commit, strips the git
history and any large binaries, and writes a `SOURCE.md` next to the code
saying where it came from and under what licence. Re-running updates the
snapshot and the recorded commit.

    python scripts/vendor_audit_fixtures.py

Both fixtures are MIT-licensed public repositories, redistributed here with
their licence text intact. They are audit *input*: the point is to have real
third-party code to point the checks at, not a judgement of anyone's work.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEST = Path("demo/audit_fixtures")

#: Keep the snapshots small: source, licence and readme only.
KEEP_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".csv"}
MAX_FILE_BYTES = 200_000


@dataclass(frozen=True)
class Fixture:
    slug: str
    url: str
    directory: str
    why: str
    keep: tuple[str, ...] = ()


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        slug="AbdullahKaddoura/acwr",
        url="https://github.com/AbdullahKaddoura/acwr",
        directory="acwr",
        why=(
            "A small, dependency-free implementation of acute:chronic workload ratio "
            "with both rolling and EWMA methods. The clearest possible target for the "
            "ACWR coupling checks."
        ),
    ),
    Fixture(
        slug="mdana5/injury-prediction",
        url="https://github.com/mdana5/injury-prediction",
        directory="injury_prediction",
        why=(
            "A full injury-prediction pipeline: feature extraction, windowing, an LSTM "
            "and a train/test split. Exercises the lookahead, label-bleed and "
            "cross-validation checks on code nobody wrote for this audit."
        ),
    ),
)


def vendor(fixture: Fixture, dest_root: Path) -> dict[str, str]:
    target = dest_root / fixture.directory
    with tempfile.TemporaryDirectory() as workspace:
        clone = Path(workspace) / fixture.directory
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", fixture.url, str(clone)],
            check=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

        copied = 0
        for path in sorted(clone.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix.lower() not in KEEP_SUFFIXES and path.name != "LICENSE":
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            relative = path.relative_to(clone)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied += 1

    (target / "SOURCE.md").write_text(
        "\n".join(
            [
                f"# {fixture.slug}",
                "",
                f"Vendored from <{fixture.url}> at commit `{commit}`.",
                "",
                f"**Why it is here:** {fixture.why}",
                "",
                "This is a snapshot of a public MIT-licensed repository, redistributed "
                "under that licence (see `LICENSE` in this directory). It is included as "
                "audit input so `ruck audit demo/` has real third-party code to read "
                "offline. Findings reported against it describe patterns in a snapshot of "
                "the code, not the competence of its authors — the same patterns appear "
                "in most published workload-modelling code, which is the point the audit "
                "is making.",
                "",
                "Nothing here is imported or executed. `ruck audit` parses it with `ast`.",
                "",
                "Refresh with `python scripts/vendor_audit_fixtures.py`.",
                "",
            ]
        )
    )
    return {"slug": fixture.slug, "url": fixture.url, "commit": commit, "files": str(copied)}


def main() -> int:
    dest = DEST
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for fixture in FIXTURES:
        try:
            record = vendor(fixture, dest)
        except subprocess.CalledProcessError as exc:
            print(f"failed to vendor {fixture.slug}: {exc}", file=sys.stderr)
            return 1
        print(f"{record['slug']}  {record['commit'][:12]}  {record['files']} files")
        manifest.append(record)
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
