"""Markdown renderer for `ruck audit`.

Two rules the renderer enforces rather than trusts. Every printed finding has a
``file:line``; anything without one is held back and counted in a footer, so a
citation-free claim can never reach the page. And the ACWR section is always
printed, findings or none, because "we looked and the ratio is uncoupled" is
the answer a performance department needs in writing.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ruck.audit.checks import CHECK_IDS
from ruck.audit.models import AuditResult, Category, Finding, Severity, Source
from ruck.audit.runner import RepoFacts

SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
)

CATEGORY_TITLES: dict[Category, str] = {
    Category.ACWR_COUPLING: "Acute:chronic ratio",
    Category.LOOKAHEAD: "Lookahead",
    Category.LABEL_BLEED: "Label bleed",
    Category.SURVIVORSHIP: "Survivorship",
    Category.UNPURGED_CV: "Unpurged cross-validation",
    Category.PREPROCESSING: "Preprocessing leakage",
    Category.OTHER: "Other",
}

#: The GPS metrics a purge list is written against when the repo names none.
DEFAULT_GPS_COLUMNS: tuple[str, ...] = (
    "total_distance",
    "high_speed_running",
    "sprint_distance",
    "player_load",
    "accelerations",
    "decelerations",
    "session_rpe",
    "duration_minutes",
)


def render(result: AuditResult, facts: RepoFacts, *, generated_at: str | None = None) -> str:
    """Render the whole report."""
    stamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    cited = [f for f in result.sorted_findings() if f.file and f.line]
    uncited = result.uncited()

    lines: list[str] = []
    lines += _header(result, stamp)
    lines += _summary(result, cited)
    lines += _acwr_section(cited)
    lines += _findings_section(cited)
    lines += _purge_list(cited, facts)
    lines += _footer(result, uncited)
    return "\n".join(lines).rstrip() + "\n"


def _header(result: AuditResult, stamp: str) -> list[str]:
    engine = "ruck native checks"
    if result.hindsight_used:
        engine += " + hindsight"
    return [
        "# ruck audit",
        "",
        f"**Target:** `{result.root}`  ",
        f"**Scanned:** {result.files_scanned} Python file"
        f"{'' if result.files_scanned == 1 else 's'} · {len(CHECK_IDS)} checks · {engine}  ",
        f"**Generated:** {stamp}",
        "",
        "The target directory is only ever read. Files are parsed with `ast`; nothing "
        "in them is imported or executed.",
        "",
    ]


def _summary(result: AuditResult, cited: list[Finding]) -> list[str]:
    lines = ["## Summary", ""]
    if not cited:
        lines += [
            "**No findings.** Every check ran and none of them fired. That is a "
            "statement about the patterns below, not a guarantee: a leak expressed in "
            "SQL, in a notebook cell that was deleted, or in how the data was collected "
            "is outside what a static reader can see.",
            "",
        ]
        return lines

    lines.append(f"**{len(cited)} finding{'' if len(cited) == 1 else 's'}.**")
    lines += ["", "| Severity | Count |", "| --- | ---: |"]
    counts = result.counts_by_severity()
    for severity in SEVERITY_ORDER:
        count = counts.get(severity.value, 0)
        if count:
            lines.append(f"| {severity.value} | {count} |")
    lines += ["", "| Category | Count |", "| --- | ---: |"]
    for category, count in sorted(
        result.counts_by_category().items(), key=lambda kv: (-kv[1], kv[0])
    ):
        title = CATEGORY_TITLES.get(Category(category), category)
        lines.append(f"| {title} | {count} |")
    lines.append("")
    return lines


def _acwr_section(cited: list[Finding]) -> list[str]:
    findings = [f for f in cited if f.category == Category.ACWR_COUPLING]
    lines = [
        "## Acute:chronic workload ratio",
        "",
        "The rugby-specific section. An ACWR divides a recent window of load by a "
        "longer one over the same series. In the standard formulation the acute window "
        "sits inside the chronic window, so writing the chronic block as the acute week "
        "plus the three before it, `ACWR = A / ((A + B) / 4) = 4A / (A + B)`. That is a "
        "bounded, increasing function of acute load on its own: it goes up after a hard "
        "week whether or not hard weeks cause injuries. A model fitted on it inherits "
        "the relationship from the algebra.",
        "",
    ]
    if not findings:
        lines += [
            "**Nothing found.** No coupled ratio, uncapped denominator or sweet-spot "
            "binning was detected in this repository. If it computes an ACWR at all, it "
            "does so from windows that do not overlap.",
            "",
        ]
        return lines

    lines.append(
        f"**{len(findings)} finding{'' if len(findings) == 1 else 's'} in this section.** "
        "Each one is written up in full below."
    )
    lines += ["", "| Check | Where | Severity | What |", "| --- | --- | --- | --- |"]
    for finding in findings:
        lines.append(
            f"| {finding.check_id} | `{finding.citation()}` | {finding.severity.value} | "
            f"{finding.title} |"
        )
    lines += [
        "",
        "The single most useful thing to do with a coupled ratio is to recompute it "
        "uncoupled and refit. If the association with injury survives, it was real; if "
        "it disappears, the model was reading its own arithmetic.",
        "",
    ]
    return lines


def _findings_section(cited: list[Finding]) -> list[str]:
    lines = ["## Findings", ""]
    if not cited:
        lines += ["None.", ""]
        return lines
    for index, finding in enumerate(cited, start=1):
        lines += _one_finding(index, finding)
    return lines


def _one_finding(index: int, finding: Finding) -> list[str]:
    detected = "ruck" if finding.source is Source.NATIVE else "hindsight"
    category = CATEGORY_TITLES.get(finding.category, finding.category.value)
    lines = [
        f"### {index}. {finding.check_id} — {finding.title}",
        "",
        f"**{finding.severity.value}** · {category} · confidence "
        f"{finding.confidence.value} · detected by {detected}",
        "",
        f"`{finding.citation()}`",
        "",
    ]
    if finding.snippet:
        lines += ["```python", finding.snippet, "```", ""]
    if finding.mechanism:
        lines += ["**Why this matters**", "", finding.mechanism, ""]
    if finding.fix:
        lines += ["**Fix**", "", finding.fix, ""]
    return lines


def _purge_list(cited: list[Finding], facts: RepoFacts) -> list[str]:
    workload = sorted(facts.workload_columns) or list(DEFAULT_GPS_COLUMNS)
    labels = sorted(facts.label_columns)
    athletes = sorted(facts.athlete_columns)
    longest_window = _longest_window(cited)

    lines = [
        "## Purge list — GPS workload data",
        "",
        "Written for a squad's GPS and session-load export: one row per athlete per "
        "session or per day, columns for distance, high-speed running, player load, "
        "accelerations and session RPE, and an availability or injury record alongside "
        "it. Work down the list; each item is a thing to remove or change before "
        "refitting.",
        "",
    ]

    lines += [
        "### 1. Purge the current observation from every feature window",
        "",
        "A session's own load must not appear in a feature predicting that session's "
        "outcome. Shift by one observation before every window:",
        "",
        "```python",
        "loads = panel.sort_values(['"
        + (athletes[0] if athletes else "player_id")
        + "', '"
        + (sorted(facts.time_columns)[0] if facts.time_columns else "date")
        + "'])",
        "past = loads.groupby('"
        + (athletes[0] if athletes else "player_id")
        + "')['"
        + (workload[0] if workload else "total_distance")
        + "'].shift(1)",
        "acute = past.rolling(7, min_periods=7).sum()",
        "```",
        "",
        "Columns this applies to in this repository: "
        + ", ".join(f"`{name}`" for name in workload[:12])
        + ("" if len(workload) <= 12 else f", and {len(workload) - 12} more")
        + ".",
        "",
    ]

    lines += [
        "### 2. Purge the acute window out of the chronic window",
        "",
        "Compute chronic load from the period *before* the acute one so the two share no sessions:",
        "",
        "```python",
        "acute   = past.rolling(7,  min_periods=7).sum()",
        "chronic = past.shift(7).rolling(21, min_periods=21).sum() / 3   # weekly equivalent",
        "acwr    = acute / chronic",
        "```",
        "",
        "Report the coupled and uncoupled ratios side by side for one season before "
        "choosing. A relationship that only exists in the coupled version is arithmetic.",
        "",
    ]

    lines += [
        "### 3. Purge rows whose windows are not yet full",
        "",
        "The first four weeks of every athlete's record have an incomplete chronic "
        "window. Passing `min_periods` and then dropping those rows is honest; letting "
        "pandas emit a partial mean is not, and clipping the resulting extreme ratios "
        "keeps a fabricated number in the sample. Expect to lose the opening 27 "
        "athlete-days of each record and one window after every gap in collection.",
        "",
    ]

    lines += [
        "### 4. Purge the outcome and everything built from it out of the feature matrix",
        "",
        (
            "Outcome columns found in this repository: "
            + ", ".join(f"`{name}`" for name in labels[:10])
            + "."
            if labels
            else "No outcome column was named literally in this repository; identify it "
            "by hand before applying this step."
        ),
        "",
        "Keep out of `X`: the outcome itself, anything derived from it "
        "(`days_since_injury`, `injury_count`, `availability`, `games_missed`) unless "
        "recomputed as of each prediction date, and any column whose value is only "
        "known once the season is over (appearances, total minutes, sessions "
        "completed). Build the feature list explicitly and assert the outcome is absent, "
        "rather than dropping by name afterwards.",
        "",
    ]

    lines += [
        "### 5. Purge the fold boundary",
        "",
        f"Split forward in time with an embargo of at least {longest_window} days — the "
        "longest feature window in this repository — so no test row shares data with a "
        "training row, and group by athlete so the same player is not on both sides:",
        "",
        "```python",
        "splitter = TimeSeriesSplit(n_splits=5, gap=" + str(longest_window) + ")",
        "#   plus: hold out whole athletes as well as later dates, and pass",
        "#   groups=panel['" + (athletes[0] if athletes else "player_id") + "']",
        "```",
        "",
    ]

    lines += [
        "### 6. Purge non-random missingness rather than filling it",
        "",
        "Missing GPS clusters around injury, illness and rehab, which is exactly the "
        "period being modelled. Do not backfill and do not interpolate across a gap. "
        "Carry the last value forward only if that is defensible for the metric, "
        "otherwise leave the gap, add a missing-indicator column, and report how many "
        "athlete-days were affected and how close they sat to recorded injuries.",
        "",
    ]
    return lines


def _longest_window(cited: list[Finding]) -> int:
    """The largest window any finding mentions, for the embargo recommendation."""
    windows = [28]
    for finding in cited:
        for key in ("window", "chronic_window", "acute_window"):
            value = finding.evidence.get(key)
            if isinstance(value, (int, float)):
                windows.append(int(value))
    return max(windows)


def _footer(result: AuditResult, uncited: list[Finding]) -> list[str]:
    lines = ["## Notes", ""]
    if result.hindsight_used:
        lines.append(
            f"Hindsight was run as `{result.hindsight_command}` and its findings "
            "are merged above, marked as detected by hindsight. Where both "
            "engines flagged the same line, ruck's wording was kept."
        )
    else:
        lines.append(
            "Hindsight was not available, so only ruck's native checks ran. "
            "Install it and re-run for the wider trading-research taxonomy on "
            "top of these."
        )
    lines.append("")
    if uncited:
        lines += [
            f"{len(uncited)} finding(s) arrived without a file and line and were not "
            "printed. A leakage claim you cannot point at is not a finding.",
            "",
        ]
    if result.errors:
        lines += ["### Files that could not be read", ""]
        for error in result.errors[:20]:
            lines.append(f"- {error}")
        if len(result.errors) > 20:
            lines.append(f"- ... and {len(result.errors) - 20} more")
        lines.append("")
    lines += [
        "### What this audit cannot see",
        "",
        "- Leakage inside a SQL query or a data warehouse view that arrives as a finished table.",
        "- A window that is correct in code but wrong because the export's timestamps "
        "are session start rather than session end.",
        "- Anything in a notebook whose cells were run out of order.",
        "- Whether the injury records themselves were entered retrospectively, which "
        "makes even a correct pipeline unusable.",
        "",
    ]
    return lines
