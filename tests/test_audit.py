from __future__ import annotations

import json
from pathlib import Path

import pytest

from ruck.audit import audit, audit_repo
from ruck.audit.checks import CHECK_IDS
from ruck.audit.hindsight import HindsightRun
from ruck.audit.hindsight import run as run_hindsight
from ruck.audit.models import Category, Confidence, Finding, Severity, Source

FIXTURES = Path(__file__).parent / "fixtures"
LEAKY = FIXTURES / "leaky_workload"
VENDORED = Path("demo/audit_fixtures")


@pytest.fixture(scope="module")
def leaky():
    result, facts = audit_repo(LEAKY, use_hindsight=False)
    return result, facts


@pytest.fixture(scope="module")
def by_check(leaky):
    result, _ = leaky
    grouped: dict[str, list[Finding]] = {}
    for finding in result.findings:
        grouped.setdefault(finding.check_id, []).append(finding)
    return grouped


# -- coverage ------------------------------------------------------------


#: AC002 is the list-slice form of AC001 and needs pure-Python code; it has its
#: own test against the vendored library below.
PANDAS_FIXTURE_CHECKS = tuple(sorted(set(CHECK_IDS) - {"AC002"}))


@pytest.mark.parametrize("check_id", PANDAS_FIXTURE_CHECKS)
def test_every_check_fires_on_the_known_bad_fixture(by_check, check_id):
    assert check_id in by_check, f"{check_id} did not fire on tests/fixtures/leaky_workload"


def test_no_finding_is_reported_without_a_citation(leaky):
    result, _ = leaky
    assert result.uncited() == []
    for finding in result.findings:
        assert finding.file
        assert finding.line > 0
        assert ":" in finding.citation()


def test_every_finding_explains_itself_and_offers_a_fix(leaky):
    result, _ = leaky
    for finding in result.findings:
        assert len(finding.mechanism) > 60, finding.check_id
        assert len(finding.fix) > 20, finding.check_id
        assert finding.snippet


def test_findings_land_on_the_line_they_describe(by_check):
    source = (LEAKY / "pipeline.py").read_text().splitlines()
    assert "shift(-1)" in source[by_check["LA001"][0].line - 1]
    assert "center=True" in source[by_check["LA002"][0].line - 1]
    assert "bfill" in source[by_check["LA003"][0].line - 1]
    assert "shuffle=True" in source[by_check["CV001"][0].line - 1]


def test_the_audit_never_writes_to_the_target():
    before = {p: p.stat().st_mtime_ns for p in LEAKY.rglob("*") if p.is_file()}
    audit_repo(LEAKY, use_hindsight=False)
    after = {p: p.stat().st_mtime_ns for p in LEAKY.rglob("*") if p.is_file()}
    assert before == after


def test_a_syntax_error_is_reported_not_raised(tmp_path):
    (tmp_path / "broken.py").write_text("def oops(:\n    pass\n")
    (tmp_path / "fine.py").write_text("x = 1\n")
    result, _ = audit_repo(tmp_path, use_hindsight=False)
    assert result.files_scanned == 1
    assert any("broken.py" in error for error in result.errors)


def test_only_filters_to_the_requested_checks():
    result, _ = audit_repo(LEAKY, use_hindsight=False, only=("AC001",))
    assert result.findings
    assert {f.check_id for f in result.findings} == {"AC001"}


def test_a_clean_file_produces_nothing(tmp_path):
    (tmp_path / "clean.py").write_text(
        "import pandas as pd\n\n\ndef load(path):\n    return pd.read_csv(path)\n"
    )
    result, _ = audit_repo(tmp_path, use_hindsight=False)
    assert result.findings == []


# -- the ACWR section ----------------------------------------------------


def test_coupled_pandas_ratio_is_critical(by_check):
    finding = by_check["AC001"][0]
    assert finding.severity is Severity.CRITICAL
    assert finding.category is Category.ACWR_COUPLING
    assert finding.evidence["acute_window"] == 7
    assert finding.evidence["chronic_window"] == 28
    assert finding.evidence["shared_observations"] == 7


def test_the_coupling_explanation_states_the_algebra(by_check):
    mechanism = by_check["AC001"][0].mechanism
    assert "7 observations are counted in both" in mechanism
    assert "A + B" in mechanism


def test_an_uncoupled_ratio_is_not_flagged(tmp_path):
    (tmp_path / "good.py").write_text(
        "import pandas as pd\n\n\n"
        "def features(panel):\n"
        "    past = panel['total_distance'].shift(1)\n"
        "    acute = past.rolling(7, min_periods=7).sum()\n"
        "    chronic = past.shift(7).rolling(21, min_periods=21).sum() / 3\n"
        "    acwr = acute / chronic\n"
        "    return acwr\n"
    )
    result, _ = audit_repo(tmp_path, use_hindsight=False)
    assert [f.check_id for f in result.findings if f.check_id.startswith("AC")] == []


def test_a_guarded_denominator_is_not_flagged(tmp_path):
    (tmp_path / "guarded.py").write_text(
        "import pandas as pd\n\n\n"
        "def features(panel):\n"
        "    past = panel['load'].shift(1)\n"
        "    acute = past.rolling(7, min_periods=7).sum()\n"
        "    chronic = past.shift(7).rolling(21, min_periods=21).sum()\n"
        "    return acute / chronic\n"
    )
    result, _ = audit_repo(tmp_path, use_hindsight=False)
    assert "AC004" not in {f.check_id for f in result.findings}


def test_the_list_slice_form_of_coupling_is_caught(tmp_path):
    (tmp_path / "pure.py").write_text(
        "def rolling_acwr(loads, acute=7, chronic=28):\n"
        "    series = [float(x) for x in loads]\n"
        "    out = []\n"
        "    for i in range(len(series)):\n"
        "        acute_window = series[max(0, i - acute + 1) : i + 1]\n"
        "        chronic_window = series[max(0, i - chronic + 1) : i + 1]\n"
        "        acute_mean = sum(acute_window) / len(acute_window)\n"
        "        chronic_mean = sum(chronic_window) / len(chronic_window)\n"
        "        out.append(acute_mean / chronic_mean)\n"
        "    return out\n"
    )
    result, _ = audit_repo(tmp_path, use_hindsight=False)
    finding = next(f for f in result.findings if f.check_id == "AC002")
    assert finding.severity is Severity.CRITICAL
    assert finding.evidence["acute_window"] == "acute"
    assert finding.evidence["chronic_window"] == "chronic"


def test_slices_that_do_not_overlap_are_left_alone(tmp_path):
    (tmp_path / "ok.py").write_text(
        "def uncoupled(series, acute=7, chronic=28):\n"
        "    out = []\n"
        "    for i in range(len(series)):\n"
        "        acute_window = series[max(0, i - acute + 1) : i + 1]\n"
        "        chronic_window = series[max(0, i - chronic + 1) : i - acute + 1]\n"
        "        a = sum(acute_window) / len(acute_window)\n"
        "        c = sum(chronic_window) / len(chronic_window)\n"
        "        out.append(a / c)\n"
        "    return out\n"
    )
    result, _ = audit_repo(tmp_path, use_hindsight=False)
    assert "AC002" not in {f.check_id for f in result.findings}


def test_sweet_spot_bins_are_collapsed_to_one_finding_per_file(by_check):
    assert len(by_check["AC005"]) == 1


# -- report rendering ----------------------------------------------------


@pytest.fixture(scope="module")
def markdown(leaky):
    from ruck.audit.report import render

    result, facts = leaky
    return render(result, facts, generated_at="2024-01-01 00:00 UTC")


def test_report_cites_every_finding_it_prints(markdown, leaky):
    result, _ = leaky
    for finding in result.findings:
        assert f"`{finding.citation()}`" in markdown


def test_report_has_a_dedicated_acwr_section(markdown):
    assert "## Acute:chronic workload ratio" in markdown
    assert "4A / (A + B)" in markdown


def test_the_acwr_section_is_present_even_with_nothing_to_report(tmp_path):
    (tmp_path / "clean.py").write_text("x = 1\n")
    _, text = audit(tmp_path, use_hindsight=False)
    assert "## Acute:chronic workload ratio" in text
    assert "**Nothing found.**" in text


def test_report_has_a_gps_purge_list(markdown):
    assert "## Purge list — GPS workload data" in markdown
    assert "Purge the acute window out of the chronic window" in markdown
    assert "min_periods" in markdown


def test_purge_list_names_the_columns_it_found(markdown):
    assert "`total_distance`" in markdown
    assert "`player_load`" in markdown


def test_report_says_what_it_cannot_see(markdown):
    assert "What this audit cannot see" in markdown


def test_report_is_written_to_disk(tmp_path):
    out = tmp_path / "audit.md"
    result, text = audit(LEAKY, out=out, use_hindsight=False)
    assert out.read_text() == text
    assert result.findings


# -- the hindsight wrapper -----------------------------------------------


def test_a_missing_hindsight_is_not_an_error(tmp_path):
    run = run_hindsight(tmp_path, executable="definitely-not-a-real-binary")
    assert isinstance(run, HindsightRun)
    assert run.used is False
    assert run.available is False
    assert "not on PATH" in (run.error or "")


def test_the_wrapper_tries_audit_before_run(monkeypatch, tmp_path):
    """The spec says `hindsight run`; the CLI implements `hindsight audit`."""
    import subprocess

    tried: list[str] = []

    class Completed:
        returncode = 0
        stdout = json.dumps({"findings": [], "files_scanned": 3})
        stderr = ""

    def fake_run(command, **kwargs):
        tried.append(command[1])
        return Completed()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("ruck.audit.hindsight.find_cli", lambda explicit=None: "hindsight")
    run = run_hindsight(tmp_path)
    assert tried == ["audit"]
    assert run.used is True
    assert run.files_scanned == 3


def test_the_wrapper_falls_back_to_run_when_audit_is_unknown(monkeypatch, tmp_path):
    import subprocess

    tried: list[str] = []

    class Completed:
        def __init__(self, code, stdout):
            self.returncode = code
            self.stdout = stdout
            self.stderr = "unknown command"

    def fake_run(command, **kwargs):
        tried.append(command[1])
        if command[1] == "audit":
            return Completed(2, "")
        return Completed(1, json.dumps({"findings": [], "files_scanned": 1}))

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("ruck.audit.hindsight.find_cli", lambda explicit=None: "hindsight")
    run = run_hindsight(tmp_path)
    assert tried == ["audit", "run"]
    assert run.used is True


def test_hindsight_findings_are_adapted_into_the_shared_model():
    finding = Finding.from_hindsight(
        {
            "check_id": "LA004",
            "category": "lookahead",
            "title": "future data in a feature",
            "severity": "critical",
            "confidence": "high",
            "file": "strategy.py",
            "line": 17,
            "snippet": "x = y.shift(-1)",
            "mechanism": "reads the future",
            "fix": "do not",
        }
    )
    assert finding.source is Source.HINDSIGHT
    assert finding.category is Category.LOOKAHEAD
    assert finding.severity is Severity.CRITICAL
    assert finding.citation() == "strategy.py:17"


def test_an_unknown_hindsight_category_lands_in_other():
    finding = Finding.from_hindsight({"category": "cost-fantasy", "file": "a.py", "line": 1})
    assert finding.category is Category.OTHER
    assert finding.confidence is Confidence.MEDIUM


# -- the vendored real-world fixtures ------------------------------------


@pytest.mark.skipif(not VENDORED.exists(), reason="vendored fixtures not present")
def test_the_vendored_acwr_library_is_caught():
    result, _ = audit_repo(VENDORED, use_hindsight=False)
    ids = {f.check_id for f in result.findings}
    assert "AC002" in ids, "the pure-Python coupled ratio should be found"
    assert "AC003" in ids, "the EWMA coupling should be found"
    assert result.uncited() == []


@pytest.mark.skipif(not VENDORED.exists(), reason="vendored fixtures not present")
def test_the_vendored_fixtures_do_not_drown_the_report_in_noise():
    result, _ = audit_repo(VENDORED, use_hindsight=False)
    # Real third-party code: a handful of defensible findings, not one per line.
    assert 0 < len(result.findings) <= 20
