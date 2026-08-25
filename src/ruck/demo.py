"""`ruck demo`: both pipelines, offline, on bundled data.

Produces one PDF and one `audit.md` from a fresh clone with no network. Three
things are bundled and they are kept strictly apart:

* a **generated** phase-play corpus and the model trained on it, which is what
  the surprisal report is built from — labelled SYNTHETIC on every page;
* a **real** Six Nations match (Italy v England, 3 February 2024) cached from
  ESPN, ingested live by the same loader and shown as an appendix, so the
  report demonstrates the real ingest path without mixing measured events into
  generated analysis;
* two **real** public injury-model repositories, audited as they are.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ruck.analyze import analyze, load_model
from ruck.audit import audit
from ruck.ingest.espn import load_espn
from ruck.ingest.http import Fetcher
from ruck.report.surprisal import format_clock

PACKAGE_DEMO = Path(__file__).parent / "_demo"
REPO_DEMO = Path("demo")

#: The bundled real match: Six Nations 2024, Italy 24-27 England.
ESPN_EVENT_ID = "597378"
ESPN_LEAGUE = "six_nations"


def demo_root() -> Path:
    """The bundled demo directory, whether running from the repo or a wheel."""
    for candidate in (REPO_DEMO, PACKAGE_DEMO):
        if (candidate / "audit_fixtures").exists():
            return candidate
    raise FileNotFoundError(
        "bundled demo data not found. From a clone, run `ruck demo` at the repo root."
    )


def run_demo(*, out_dir: str | Path = "runs/demo", device: str = "auto") -> dict[str, Path]:
    """Run both pipelines and return the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = demo_root()

    corpus_path = next(
        (
            root / "model" / name
            for name in ("corpus.csv.gz", "corpus.csv")
            if (root / "model" / name).exists()
        ),
        None,
    )
    if corpus_path is None:
        raise FileNotFoundError(
            f"{root / 'model'} has no corpus. Run `python scripts/train_demo_model.py` "
            "to build the bundled model and corpus."
        )

    match_path = _write_demo_match(corpus_path, out_dir / "demo_match.csv")
    model = load_model(root / "model")
    real_check = _real_match_appendix(root)

    print("analyzing the bundled synthetic match ...")
    analysis = analyze(
        match_path,
        out=out_dir / "report.pdf",
        clip_csv=out_dir / "clips.csv",
        model=model,
        reference=corpus_path,
        device=device,
    )
    analysis.context.real_check = real_check
    # Re-render now that the appendix is attached.
    from ruck.report.render import render_report

    render_report(
        scored=analysis.scored,
        timeline=analysis.timeline,
        highlights=analysis.highlights,
        match_shape=analysis.match_shape,
        reference_shape=analysis.reference_shape,
        context=analysis.context,
        out_path=out_dir / "report.pdf",
    )
    print(f"  {analysis.load_report.summary().splitlines()[0]}")
    print(f"  {len(analysis.highlights)} clips, {len(analysis.scored)} events")

    print("auditing the bundled injury-model repositories ...")
    result, _ = audit(root / "audit_fixtures", out=out_dir / "audit.md")
    cited = [f for f in result.findings if f.file and f.line]
    print(
        f"  {result.files_scanned} files, {len(cited)} findings: "
        + (", ".join(f"{k} {v}" for k, v in result.counts_by_severity().items()) or "none")
    )

    return {
        "match": match_path,
        "pdf": out_dir / "report.pdf",
        "clips": out_dir / "clips.csv",
        "audit": out_dir / "audit.md",
    }


def _write_demo_match(corpus_path: Path, out_path: Path) -> Path:
    """Pull one match out of the bundled corpus and write it as a standalone CSV."""
    corpus = pd.read_csv(corpus_path)
    match_id = sorted(corpus["match_id"].unique())[0]
    match = corpus[corpus["match_id"] == match_id]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    match.to_csv(out_path, index=False)
    return out_path


def _real_match_appendix(root: Path) -> dict[str, object]:
    """Ingest the cached Six Nations match and render it as a scoring timeline."""
    fetcher = Fetcher(cache_dir=root / "espn_cache", offline=True)
    events, report = load_espn(ESPN_EVENT_ID, ESPN_LEAGUE, fetcher)
    scoring = events[events["outcome"] == "points"]

    rows = [
        [
            format_clock(float(row["t"])),
            str(row["team"]),
            str(row["player"]),
            str(row["type"]).replace("_", " "),
        ]
        for _, row in scoring.iterrows()
    ]
    competition = str(events["competition"].iloc[0])
    season = str(events["season"].iloc[0])
    teams = sorted(events["team"].dropna().unique())
    return {
        "label": f"Appendix — real match: {teams[0]} v {teams[1]}, {competition} {season}",
        "note": (
            f"Ingested offline from a cached ESPN payload by `ruck.ingest.espn`: "
            f"{report.events_out} events, {len(rows)} of them scoring. This is real, "
            "measured data. ESPN's rugby feed carries scoring, card and substitution "
            "events with a match clock and no pitch coordinates, so it supports the "
            "timeline below but not phase-play analysis — which is why the report above "
            "is built from generated data instead."
        ),
        "columns": ["Clock", "Team", "Player", "Event"],
        "rows": rows,
    }
