"""The `ruck analyze` pipeline: events in, PDF and clip list out."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ruck.ingest import Fetcher, LoadReport, load
from ruck.ingest.generic_csv import load_csv
from ruck.lm.model import PhaseLM
from ruck.report.render import ReportContext, render_report
from ruck.report.shape import ShapeSummary, competition_reference, team_shape
from ruck.report.surprisal import Highlight, event_surprisal, phase_timeline, top_sequences
from ruck.report.timestamps import write_csv as write_clip_csv
from ruck.tokenizer import Tokenizer, Vocab

#: Where `ruck demo` and a default `ruck analyze` look for the shipped model.
#: In a wheel the demo data is packaged inside `ruck/_demo`; in an editable
#: install it is still up at the repo root, so that is tried too. Without the
#: third candidate the command works from the repo root and nowhere else,
#: which is a confusing way for a bundled model to behave.
BUNDLED_MODEL_DIR = Path(__file__).parent / "_demo" / "model"
SOURCE_TREE_MODEL_DIR = Path(__file__).resolve().parents[2] / "demo" / "model"
REPO_MODEL_DIR = Path("demo/model")


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained model is available to score with."""


@dataclass
class LoadedModel:
    """A tokenizer, a model, and the metrics that describe them."""

    tokenizer: Tokenizer
    model: PhaseLM
    metrics: dict[str, object] = field(default_factory=dict)
    directory: Path | None = None

    def trained_on_synthetic(self) -> bool:
        """True when the shipped metrics say the training corpus was generated.

        A report has to say this even when the match being analysed is real,
        because it is the model's surprisal that every number on the timeline
        comes from.
        """
        data = self.metrics.get("data")
        return bool(isinstance(data, dict) and data.get("synthetic"))

    def model_card(self) -> dict[str, object]:
        """Flatten the metrics file into the rows the PDF's model card prints."""
        card: dict[str, object] = {
            "n_params": self.model.n_params(),
            "vocab_size": len(self.tokenizer.vocab),
        }
        data = self.metrics.get("data")
        if isinstance(data, dict):
            card["train_events"] = data.get("train_events")
            card["train_matches"] = data.get("train_matches")
        perplexity = self.metrics.get("perplexity")
        if isinstance(perplexity, dict):
            for name, key in (
                ("transformer", "ppl_transformer"),
                ("bigram", "ppl_bigram"),
                ("unigram", "ppl_unigram"),
            ):
                entry = perplexity.get(name)
                if isinstance(entry, dict):
                    card[key] = entry.get("perplexity")
        downstream = self.metrics.get("downstream")
        if isinstance(downstream, dict):
            for name, result in (downstream.get("results") or {}).items():
                if isinstance(result, dict):
                    card[f"downstream_{name}"] = (
                        f"AUC {result.get('roc_auc', float('nan')):.3f}  "
                        f"AP {result.get('average_precision', float('nan')):.3f}"
                    )
        return card


def find_model_dir(explicit: str | Path | None = None) -> Path:
    """Locate a model directory: the one given, the repo's, or the packaged one."""
    candidates = (
        [Path(explicit)] if explicit else [REPO_MODEL_DIR, BUNDLED_MODEL_DIR, SOURCE_TREE_MODEL_DIR]
    )
    for candidate in candidates:
        if (candidate / "phase_lm.pt").exists() and (candidate / "vocab.json").exists():
            return candidate
    raise ModelNotFoundError(
        "no trained model found. Run `python scripts/train_demo_model.py` to build the "
        "bundled one, or point --model at a directory holding phase_lm.pt and vocab.json."
    )


def load_model(directory: str | Path | None = None) -> LoadedModel:
    """Load a tokenizer, model and metrics from a model directory."""
    path = find_model_dir(directory)
    tokenizer = Tokenizer(Vocab.load(path / "vocab.json"))
    model = PhaseLM.load(path / "phase_lm.pt")
    metrics: dict[str, object] = {}
    metrics_path = path / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    return LoadedModel(tokenizer=tokenizer, model=model, metrics=metrics, directory=path)


@dataclass
class Analysis:
    """Everything `ruck analyze` produced, so callers can assert on it."""

    events: pd.DataFrame
    scored: pd.DataFrame
    timeline: pd.DataFrame
    highlights: list[Highlight]
    match_shape: ShapeSummary
    reference_shape: ShapeSummary | None
    context: ReportContext
    load_report: LoadReport
    pdf_path: Path | None = None
    csv_path: Path | None = None


def analyze(
    target: str | Path,
    *,
    team: str | None = None,
    out: str | Path | None = None,
    clip_csv: str | Path | None = None,
    model: str | Path | LoadedModel | None = None,
    reference: str | Path | None = None,
    device: str = "auto",
    top_n: int = 10,
    window: int = 3,
    fetcher: Fetcher | None = None,
    **load_kwargs: object,
) -> Analysis:
    """Load ``target``, score it, and write the report.

    Args:
        target: a CSV path, a RugbyPy file, or ``espn:<league>:<event_id>``.
        team: restrict the attack-shape section to one team.
        out: PDF path. No PDF is written when this is None.
        clip_csv: timestamp CSV path. Written whenever given.
        model: a model directory or an already-loaded :class:`LoadedModel`.
        reference: a corpus CSV for the competition-average comparison; defaults
            to the corpus the model was trained on, when it ships with one.
    """
    events, report = load(target, fetcher=fetcher, **load_kwargs)  # type: ignore[arg-type]
    if events.empty:
        raise ValueError(f"{target}: no events could be loaded ({report.summary()})")

    if team is not None:
        available = sorted(events["team"].dropna().astype(str).unique())
        if team not in available:
            raise ValueError(
                f"--team {team!r} does not appear in {target}. Teams in this match: "
                + (", ".join(available) if available else "none recorded")
            )

    loaded = model if isinstance(model, LoadedModel) else load_model(model)

    # How much of this match the model's vocabulary actually covers. Surprisal
    # over tokens it has never seen measures novelty of the *source*, not of
    # the rugby, and a report that does not say so is misleading.
    coverage = loaded.tokenizer.coverage(events)
    coverage_note = _coverage_note(coverage)

    scored = event_surprisal(events, loaded.tokenizer, loaded.model, device=device)
    timeline = phase_timeline(scored if team is None else scored[scored["team"] == team])
    highlights = top_sequences(
        scored if team is None else scored[scored["team"] == team], n=top_n, window=window
    )

    match_shape = team_shape(events, team)
    reference_shape, reference_note = _reference_shape(reference, loaded, events)

    context = ReportContext(
        match_id=str(events["match_id"].iloc[0]),
        competition=_first(events, "competition"),
        season=_first(events, "season"),
        teams=tuple(sorted(events["team"].dropna().unique())),
        focus_team=team,
        source_label=report.source,
        synthetic=_is_synthetic(events),
        load_notes=list(report.notes),
        model_card=loaded.model_card(),
        model_is_synthetic=loaded.trained_on_synthetic(),
        reference_note=reference_note,
        coverage=coverage,
        coverage_note=coverage_note,
    )

    analysis = Analysis(
        events=events,
        scored=scored,
        timeline=timeline,
        highlights=highlights,
        match_shape=match_shape,
        reference_shape=reference_shape,
        context=context,
        load_report=report,
    )

    if clip_csv is not None:
        analysis.csv_path = write_clip_csv(highlights, clip_csv)
    if out is not None:
        analysis.pdf_path = render_report(
            scored=scored,
            timeline=timeline,
            highlights=highlights,
            match_shape=match_shape,
            reference_shape=reference_shape,
            context=context,
            out_path=out,
        )
    return analysis


def _reference_shape(
    reference: str | Path | None, loaded: LoadedModel, events: pd.DataFrame
) -> tuple[ShapeSummary | None, str | None]:
    """Competition shape, and a note when there is none.

    The model ships with the corpus it was trained on, and falling back to it
    is only honest when the match under review came from the same place. A real
    match benchmarked against generated events, under a heading that reads
    "Competition average", is a fabricated comparison however well-labelled the
    rest of the report is — so that fallback is refused and the section says
    what to pass instead.
    """
    path: Path | None = Path(reference) if reference is not None else None
    if path is None:
        bundled = _bundled_corpus(loaded)
        if bundled is None:
            return None, None
        if not _is_synthetic(events):
            return None, (
                "No competition reference. The corpus bundled with the model is "
                "generated, and benchmarking a real match against generated events "
                "would be a fabricated comparison. Pass --reference with a corpus of "
                "your own matches to fill this section in."
            )
        path = bundled
    if not path.exists():
        return None, f"Reference corpus {path} not found."

    corpus, _ = load_csv(path)
    match_ids = set(events["match_id"].dropna().astype(str))
    reference_shape = competition_reference(corpus, exclude_matches=match_ids)
    if reference_shape.n_events == 0:
        return None, f"Reference corpus {path.name} has nothing in it but this match."
    return reference_shape, None


#: Above this share of events falling outside the model's vocabulary, the
#: surprisal ranking is not about rugby any more.
UNKNOWN_TOKEN_LIMIT = 0.10


def _coverage_note(coverage: dict[str, float]) -> str | None:
    """A warning when the model's vocabulary does not cover this match."""
    unknown = coverage.get("unk", 0.0)
    if unknown <= UNKNOWN_TOKEN_LIMIT:
        return None
    return (
        f"{unknown:.0%} of the events in this match are of a kind the model has never "
        "seen, so they fall outside its vocabulary entirely. Those events score near "
        "the maximum surprisal whatever happened in them, which means this ranking is "
        "measuring how unlike the training corpus the source is, not how unexpected "
        "the rugby was. Train on matches from the same feed (`ruck train`) before "
        "reading the ordering."
    )


def _bundled_corpus(loaded: LoadedModel) -> Path | None:
    if loaded.directory is None:
        return None
    for name in ("corpus.csv.gz", "corpus.csv"):
        if (loaded.directory / name).exists():
            return loaded.directory / name
    return None


def _first(events: pd.DataFrame, column: str) -> str | None:
    values = events[column].dropna()
    return str(values.iloc[0]) if len(values) else None


def _is_synthetic(events: pd.DataFrame) -> bool:
    """True when any event says it came from the generator."""
    for blob in events["extra"].dropna().head(50):
        try:
            if json.loads(blob).get("synthetic"):
                return True
        except (json.JSONDecodeError, AttributeError):
            continue
    competition = events["competition"].dropna()
    return bool(len(competition) and "generated" in str(competition.iloc[0]).lower())
