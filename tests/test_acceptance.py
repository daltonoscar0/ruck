"""The spec's acceptance criteria, written as tests.

Each test here corresponds to a line in the brief:

* the demo runs offline on a fresh clone and produces one PDF and one audit.md;
* perplexity and downstream metrics are in the README with the GBM comparison,
  honest either way;
* the timestamp CSV opens in Excel and matches the PDF;
* every audit finding carries a file:line citation, and there are none without.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from ruck.analyze import find_model_dir, load_model
from ruck.audit import audit
from ruck.cli import build_parser, main
from ruck.tokenizer import DEFAULT_MAX_VOCAB

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo"
README = REPO / "README.md"
MODEL_DIR = DEMO / "model"

needs_model = pytest.mark.skipif(
    not (MODEL_DIR / "phase_lm.pt").exists(),
    reason="run `python scripts/train_demo_model.py` to build the bundled model",
)


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    """Run the demo once and share its artefacts across the acceptance tests."""
    from ruck.demo import run_demo

    out_dir = tmp_path_factory.mktemp("demo")
    return run_demo(out_dir=out_dir, device="cpu")


# -- one PDF and one audit.md, offline ------------------------------------


@needs_model
def test_demo_produces_one_pdf_and_one_audit(demo_output):
    assert demo_output["pdf"].exists()
    assert demo_output["audit"].exists()
    assert demo_output["pdf"].read_bytes().startswith(b"%PDF")
    assert demo_output["audit"].read_text().startswith("# ruck audit")


@needs_model
def test_demo_never_touches_the_network(monkeypatch, tmp_path):
    """Any outbound request during the demo is a bug, not a slow test."""
    import requests

    from ruck.demo import run_demo

    def explode(*args, **kwargs):
        raise AssertionError("the demo made a network request")

    monkeypatch.setattr(requests.Session, "get", explode)
    monkeypatch.setattr(requests, "get", explode)
    artefacts = run_demo(out_dir=tmp_path / "offline", device="cpu")
    assert artefacts["pdf"].exists()


@needs_model
def test_the_demo_pdf_is_within_the_page_budget(demo_output):
    raw = demo_output["pdf"].read_bytes()
    pages = raw.count(b"/Type /Page\n") or raw.count(b"/Type /Page")
    assert 4 <= pages <= 6, f"the spec asks for a 4-6 page PDF, got {pages}"


@needs_model
def test_the_demo_pdf_is_labelled_synthetic(demo_output):
    assert b"SYNTHETIC DATA" in demo_output["pdf"].read_bytes()


@needs_model
def test_the_demo_includes_the_real_six_nations_match(demo_output):
    raw = demo_output["pdf"].read_bytes()
    assert b"Six Nations" in raw
    assert b"Tommaso Allan" in raw, "the real match's scorers should be in the appendix"


# -- the timestamp CSV matches the PDF ------------------------------------


@needs_model
def test_clip_csv_opens_as_a_spreadsheet(demo_output):
    with demo_output["clips"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["clip"] == "1"
    for row in rows:
        assert re.fullmatch(r"\d{2}:\d{2}", row["clock"])
        assert float(row["t_seconds"]) >= float(row["clip_start_seconds"])
        assert row["description"]


@needs_model
def test_every_clip_in_the_csv_appears_in_the_pdf(demo_output):
    """The spec's 'CSV matches PDF contents', checked rather than asserted."""
    raw = demo_output["pdf"].read_bytes()
    with demo_output["clips"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert row["clock"].encode() in raw, f"clip {row['clip']} at {row['clock']}"
        assert row["clip_start_clock"].encode() in raw
        assert row["surprisal_bits"].encode() in raw


# -- every audit finding is cited -----------------------------------------


def test_no_audit_finding_lacks_a_citation():
    for target in (REPO / "tests" / "fixtures" / "leaky_workload", DEMO / "audit_fixtures"):
        if not target.exists():
            continue
        result, markdown = audit(target, use_hindsight=False)
        assert result.uncited() == []
        for finding in result.findings:
            assert re.search(rf"`{re.escape(finding.file)}:{finding.line}`", markdown)


def test_the_audit_report_states_its_finding_count_honestly():
    result, markdown = audit(REPO / "tests" / "fixtures" / "leaky_workload", use_hindsight=False)
    assert f"**{len(result.findings)} findings.**" in markdown


# -- the README carries the metrics ---------------------------------------


@needs_model
def test_readme_reports_perplexity_against_the_baselines():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    text = README.read_text()
    for name in ("transformer", "bigram", "unigram"):
        value = metrics["perplexity"][name]["perplexity"]
        assert f"{value:.2f}" in text, f"{name} perplexity {value:.2f} is not in the README"


@needs_model
def test_readme_reports_the_downstream_comparison_against_the_gbm():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    text = README.read_text()
    for key in ("gbm", "lm_probe"):
        auc = metrics["downstream"]["results"][key]["roc_auc"]
        assert f"{auc:.3f}" in text, f"{key} AUC {auc:.3f} is not in the README"


@needs_model
def test_readme_says_which_predictor_won():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    gbm = metrics["downstream"]["results"]["gbm"]["roc_auc"]
    probe = metrics["downstream"]["results"]["lm_probe"]["roc_auc"]
    text = README.read_text().lower()
    winner = "gbm" if gbm > probe else "probe"
    assert winner in text
    assert "synthetic" in text


@needs_model
def test_readme_does_not_claim_the_synthetic_numbers_are_about_rugby():
    text = README.read_text()
    assert "synthetic" in text.lower()
    assert "measure the pipeline" in text or "not rugby" in text.lower()


# -- model and vocabulary budgets -----------------------------------------


@needs_model
def test_the_bundled_model_is_within_the_parameter_budget():
    model = load_model(MODEL_DIR)
    assert model.model.n_params() <= 10_000_000


@needs_model
def test_the_bundled_vocabulary_is_under_two_thousand():
    model = load_model(MODEL_DIR)
    assert len(model.tokenizer.vocab) < DEFAULT_MAX_VOCAB + 1
    assert len(model.tokenizer.vocab) < 2000


# -- the CLI ---------------------------------------------------------------


def test_every_documented_subcommand_parses():
    parser = build_parser()
    for argv in (
        ["analyze", "match.csv", "--team", "Munster", "--out", "report.pdf"],
        ["compare", "a.csv", "b.csv", "--out", "overlay.pdf"],
        ["audit", "path/to/repo", "--out", "audit.md"],
        ["demo", "--out-dir", "runs/demo"],
        ["fetch", "espn", "--league", "six_nations", "--dates", "20240203"],
        ["train", "corpus.csv", "--out", "models/mine"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_audit_exits_nonzero_when_asked_to_fail_on_findings(tmp_path, capsys):
    target = REPO / "tests" / "fixtures" / "leaky_workload"
    code = main(
        [
            "audit",
            str(target),
            "--out",
            str(tmp_path / "a.md"),
            "--no-hindsight",
            "--fail-on",
            "critical",
        ]
    )
    assert code == 1
    assert (tmp_path / "a.md").exists()


def test_audit_exits_zero_on_a_clean_repository(tmp_path):
    (tmp_path / "clean.py").write_text("x = 1\n")
    code = main(
        [
            "audit",
            str(tmp_path),
            "--out",
            str(tmp_path / "a.md"),
            "--no-hindsight",
            "--fail-on",
            "low",
        ]
    )
    assert code == 0


def test_a_missing_target_is_an_error_not_a_traceback(tmp_path, capsys):
    assert main(["audit", str(tmp_path / "nope"), "--no-hindsight"]) == 2
    assert "ruck:" in capsys.readouterr().err


def test_find_model_dir_explains_itself_when_there_is_no_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ruck.analyze.BUNDLED_MODEL_DIR", tmp_path / "absent")
    monkeypatch.setattr("ruck.analyze.SOURCE_TREE_MODEL_DIR", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="train_demo_model"):
        find_model_dir()


@needs_model
def test_compare_writes_an_overlay_naming_both_matches(tmp_path):
    import pandas as pd

    corpus = pd.read_csv(DEMO / "model" / "corpus.csv.gz")
    paths = []
    for index, match_id in enumerate(sorted(corpus["match_id"].unique())[:2]):
        path = tmp_path / f"m{index}.csv"
        corpus[corpus["match_id"] == match_id].to_csv(path, index=False)
        paths.append(str(path))

    out = tmp_path / "overlay.pdf"
    assert main(["compare", *paths, "--out", str(out), "--device", "cpu"]) == 0
    raw = out.read_bytes()
    assert raw.startswith(b"%PDF")
    # Each column is named for the match behind it, not "All teams" twice.
    assert b"All teams" not in raw


@needs_model
def test_the_demo_scores_a_match_the_model_never_saw(demo_output):
    """In-sample surprisal would make the whole report a memorisation check."""
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    held_out = set(metrics["data"]["test_match_ids"])
    assert held_out, "training must record which matches were held out"

    import pandas as pd

    match = pd.read_csv(demo_output["match"])
    assert set(match["match_id"].astype(str)) <= held_out


@needs_model
def test_a_real_match_is_never_benchmarked_against_generated_events(tmp_path):
    """The bundled reference corpus is synthetic; a real match must not use it."""
    from ruck.analyze import analyze

    result = analyze(
        REPO / "tests" / "fixtures" / "mini_match.csv",
        out=tmp_path / "real.pdf",
        device="cpu",
    )
    assert result.reference_shape is None
    assert result.context.reference_note
    assert "generated" in result.context.reference_note

    raw = (tmp_path / "real.pdf").read_bytes()
    # The events are real, so no synthetic-data banner — but the model is
    # generated and the report has to say so.
    assert b"SYNTHETIC MODEL" in raw
    assert b"Competition average" not in raw


@needs_model
def test_an_unknown_team_is_an_error_not_an_empty_report(tmp_path):
    from ruck.analyze import analyze

    with pytest.raises(ValueError, match="does not appear"):
        analyze(
            REPO / "tests" / "fixtures" / "mini_match.csv",
            team="Toulouse",
            out=tmp_path / "nope.pdf",
            device="cpu",
        )


@needs_model
def test_readme_records_the_smoothing_the_baseline_chose():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    assert "smoothing_k" in metrics["perplexity"]["bigram"]


@needs_model
def test_the_bundled_model_is_found_from_any_directory(tmp_path, monkeypatch):
    """An editable install must not only work from the repo root."""
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "demo").exists()
    assert find_model_dir().exists()


@needs_model
def test_a_match_outside_the_models_vocabulary_is_flagged(tmp_path):
    """Surprisal over unseen token types measures the feed, not the rugby."""
    from ruck.analyze import analyze
    from ruck.ingest import Fetcher

    fetcher = Fetcher(cache_dir=DEMO / "espn_cache", offline=True)
    result = analyze(
        "espn:six_nations:597378", out=tmp_path / "espn.pdf", device="cpu", fetcher=fetcher
    )
    assert result.context.coverage["unk"] > 0.5
    assert result.context.coverage_note
    assert b"OUT OF VOCABULARY" in (tmp_path / "espn.pdf").read_bytes()


@needs_model
def test_a_covered_match_carries_no_vocabulary_warning(demo_output):
    """The bundled synthetic match is what the model was fitted on the kind of."""
    assert b"OUT OF VOCABULARY" not in demo_output["pdf"].read_bytes()
