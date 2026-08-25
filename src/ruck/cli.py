"""Command line interface.

    ruck analyze match.csv --team Munster --out report.pdf
    ruck compare a.csv b.csv --out overlay.pdf
    ruck audit path/to/repo --out audit.md
    ruck demo --out-dir runs/demo
    ruck fetch espn --league six_nations --dates 20240203
    ruck train corpus.csv --out models/mine

Exit codes:
    0  ran; no finding at or above --fail-on (or --fail-on not given)
    1  ran; at least one finding at or above --fail-on
    2  could not run (bad path, no model, unreadable source)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ruck import __version__

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ruck",
        description=(
            "Rugby phase-play surprisal reports, and leakage audits of workload and injury models."
        ),
    )
    parser.add_argument("--version", action="version", version=f"ruck {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="score a match and write the report")
    analyze.add_argument("target", help="CSV, RugbyPy file, or espn:<league>:<event_id>")
    analyze.add_argument("--team", default=None, help="focus the shape section on one team")
    analyze.add_argument("--out", type=Path, default=Path("report.pdf"), help="PDF path")
    analyze.add_argument("--csv", type=Path, default=None, help="timestamp list for clipping")
    analyze.add_argument("--model", type=Path, default=None, help="model directory")
    analyze.add_argument("--reference", type=Path, default=None, help="competition corpus CSV")
    analyze.add_argument("--top", type=int, default=10, help="clips in the list")
    analyze.add_argument("--window", type=int, default=3, help="events per clip window")
    analyze.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    analyze.add_argument("--offline", action="store_true", help="never hit the network")
    analyze.add_argument("--strict", action="store_true", help="fail on a bad row")

    compare = subparsers.add_parser("compare", help="overlay two matches' surprisal")
    compare.add_argument("targets", nargs=2, help="two sources, as for analyze")
    compare.add_argument("--out", type=Path, default=Path("compare.pdf"))
    compare.add_argument("--teams", nargs=2, default=None, help="focus team for each source")
    compare.add_argument("--model", type=Path, default=None)
    compare.add_argument("--device", default="auto")
    compare.add_argument("--offline", action="store_true")

    audit = subparsers.add_parser("audit", help="audit a workload/injury-model repository")
    audit.add_argument("path", type=Path, help="repository or file to read")
    audit.add_argument("--out", type=Path, default=None, help="markdown report path")
    audit.add_argument("--no-hindsight", action="store_true", help="skip the hindsight wrapper")
    audit.add_argument("--hindsight-bin", default=None, help="path to the hindsight executable")
    audit.add_argument("--only", default=None, help="comma-separated check ids, e.g. AC001,LA002")
    audit.add_argument(
        "--fail-on",
        choices=("critical", "high", "medium", "low", "info"),
        default=None,
        help="exit 1 if any finding is at or above this severity",
    )

    demo = subparsers.add_parser("demo", help="run both pipelines offline on bundled data")
    demo.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    demo.add_argument("--device", default="auto")

    fetch = subparsers.add_parser("fetch", help="populate the download cache")
    fetch.add_argument("source", choices=("espn",))
    fetch.add_argument("--league", default="six_nations")
    fetch.add_argument("--dates", required=True, help="YYYYMMDD or YYYYMMDD-YYYYMMDD")
    fetch.add_argument("--cache-dir", type=Path, default=Path("data/raw"))

    train = subparsers.add_parser("train", help="fit a model on your own corpus")
    train.add_argument("corpus", type=Path, help="CSV in the documented event format")
    train.add_argument("--out", type=Path, default=Path("models/custom"))
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--max-vocab", type=int, default=2000)
    train.add_argument("--device", default="auto")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "analyze": _analyze,
        "compare": _compare,
        "audit": _audit,
        "demo": _demo,
        "fetch": _fetch,
        "train": _train,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR
    except (FileNotFoundError, ValueError) as exc:
        print(f"ruck: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _analyze(args: argparse.Namespace) -> int:
    from ruck.analyze import analyze
    from ruck.ingest import Fetcher

    fetcher = Fetcher(offline=args.offline) if str(args.target).startswith("espn:") else None
    result = analyze(
        args.target,
        team=args.team,
        out=args.out,
        clip_csv=args.csv,
        model=args.model,
        reference=args.reference,
        device=args.device,
        top_n=args.top,
        window=args.window,
        fetcher=fetcher,
        **({"strict": True} if args.strict else {}),
    )
    print(result.load_report.summary())
    if result.context.synthetic:
        print("  note: this source is synthetic; the report says so on every page")
    print(f"wrote {result.pdf_path}")
    if result.csv_path:
        print(f"wrote {result.csv_path} ({len(result.highlights)} clips)")
    return EXIT_OK


def _compare(args: argparse.Namespace) -> int:
    from ruck.analyze import analyze, load_model
    from ruck.ingest import Fetcher
    from ruck.report.render import render_comparison

    model = load_model(args.model)
    teams = args.teams or [None, None]
    analyses = []
    for target, team in zip(args.targets, teams, strict=True):
        fetcher = Fetcher(offline=args.offline) if str(target).startswith("espn:") else None
        analyses.append(
            analyze(target, team=team, model=model, device=args.device, fetcher=fetcher)
        )

    series = [(analysis.context.title(), analysis.timeline) for analysis in analyses]
    path = render_comparison(
        series=series,
        shapes=[analysis.match_shape for analysis in analyses],
        contexts=[analysis.context for analysis in analyses],
        out_path=args.out,
    )
    for analysis in analyses:
        print(analysis.load_report.summary())
    print(f"wrote {path}")
    return EXIT_OK


def _audit(args: argparse.Namespace) -> int:
    from ruck.audit import audit
    from ruck.audit.models import Severity

    only = tuple(c.strip() for c in args.only.split(",")) if args.only else None
    result, markdown = audit(
        args.path,
        out=args.out,
        use_hindsight=not args.no_hindsight,
        hindsight_executable=args.hindsight_bin,
        only=only,
    )
    if args.out is None:
        print(markdown)
    else:
        cited = [f for f in result.findings if f.file and f.line]
        print(
            f"{result.files_scanned} file(s) scanned, {len(cited)} finding(s): "
            + ", ".join(f"{k} {v}" for k, v in result.counts_by_severity().items())
        )
        print(f"wrote {args.out}")

    if args.fail_on:
        threshold = Severity(args.fail_on).rank
        if any(f.severity.rank <= threshold for f in result.findings):
            return EXIT_FINDINGS
    return EXIT_OK


def _demo(args: argparse.Namespace) -> int:
    from ruck.demo import run_demo

    artefacts = run_demo(out_dir=args.out_dir, device=args.device)
    for label, path in artefacts.items():
        print(f"  {label:<10} {path}")
    return EXIT_OK


def _fetch(args: argparse.Namespace) -> int:
    from ruck.ingest import Fetcher, list_matches
    from ruck.ingest.espn import load_espn

    fetcher = Fetcher(cache_dir=args.cache_dir)
    matches = list_matches(args.league, args.dates, fetcher)
    if not matches:
        print(f"no {args.league} matches on {args.dates}")
        return EXIT_OK
    for match in matches:
        _, report = load_espn(match["event_id"], args.league, fetcher)
        print(f"  {match['event_id']}  {match['name']}  —  {report.summary().splitlines()[0]}")
    print(f"cached under {args.cache_dir}")
    return EXIT_OK


def _train(args: argparse.Namespace) -> int:
    import json

    from ruck.demo_data.generate import write_csv
    from ruck.ingest import load_csv
    from ruck.lm import CountBaseline, LMConfig, TrainConfig, evaluate, split_by_match, train
    from ruck.tokenizer import Tokenizer

    events, report = load_csv(args.corpus)
    print(report.summary())
    split = split_by_match(events)
    print("  " + json.dumps(split.sizes()))

    tokenizer = Tokenizer.fit(split.train, max_size=args.max_vocab)
    print(f"  vocab {len(tokenizer.vocab)}")
    train_sequences = tokenizer.encode_events(split.train)
    val_sequences = tokenizer.encode_events(split.val)
    test_sequences = tokenizer.encode_events(split.test)

    args.out.mkdir(parents=True, exist_ok=True)
    model, history = train(
        train_sequences,
        val_sequences,
        vocab_size=len(tokenizer.vocab),
        model_config=LMConfig(vocab_size=len(tokenizer.vocab)),
        config=TrainConfig(epochs=args.epochs, device=args.device),
        checkpoint=args.out / "phase_lm.pt",
    )
    tokenizer.vocab.save(args.out / "vocab.json")
    write_csv(events, args.out / "corpus.csv")

    transformer = evaluate(model, test_sequences, device=args.device, name="transformer")
    bigram = (
        CountBaseline(len(tokenizer.vocab), order=2).fit(train_sequences).perplexity(test_sequences)
    )
    unigram = (
        CountBaseline(len(tokenizer.vocab), order=1).fit(train_sequences).perplexity(test_sequences)
    )
    metrics = {
        "data": {"synthetic": False, "source": str(args.corpus), **split.sizes()},
        "tokenizer": {"vocab_size": len(tokenizer.vocab)},
        "model": {"n_params": history.n_params},
        "perplexity": {
            r.name: {"perplexity": r.perplexity, "n_tokens": r.n_tokens}
            for r in (transformer, bigram, unigram)
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("  " + history.summary())
    for result in (transformer, bigram, unigram):
        print(f"  {result}")
    if transformer.perplexity > bigram.perplexity:
        print("  note: the transformer did not beat the bigram baseline on this corpus")
    print(f"wrote {args.out}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
