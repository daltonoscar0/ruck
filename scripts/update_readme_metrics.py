"""Rewrite the README's metrics block from `demo/model/metrics.json`.

The README quotes real numbers, so they are generated rather than typed. Run
this after `scripts/train_demo_model.py`; `tests/test_acceptance.py` checks the
two agree.

    python scripts/update_readme_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

README = Path("README.md")
METRICS = Path("demo/model/metrics.json")
START = "<!-- metrics:start -->"
END = "<!-- metrics:end -->"


def block(metrics: dict) -> str:
    data = metrics["data"]
    model = metrics["model"]
    tokenizer = metrics["tokenizer"]
    perplexity = metrics["perplexity"]
    downstream = metrics["downstream"]["results"]
    training = metrics["training"]

    gbm = downstream["gbm"]
    probe = downstream["lm_probe"]
    base = downstream["base_rate"]
    winner = "GBM on hand features" if gbm["roc_auc"] > probe["roc_auc"] else "LM probe"
    loser = "LM probe" if winner.startswith("GBM") else "GBM on hand features"

    lines = [
        START,
        "",
        "Measured on a **synthetic** corpus (see [Data](#data)): "
        f"{data['matches']} generated matches, split by match into "
        f"{data['train_matches']} train / {data['val_matches']} validation / "
        f"{data['test_matches']} test. These numbers measure the pipeline, not rugby. "
        "Reproduce with `python scripts/train_demo_model.py`.",
        "",
        "### Next-event perplexity",
        "",
        "| Model | Test perplexity |",
        "| --- | ---: |",
        f"| Causal transformer ({model['n_params']:,} params) "
        f"| **{perplexity['transformer']['perplexity']:.2f}** |",
        f"| Add-k smoothed bigram | {perplexity['bigram']['perplexity']:.2f} |",
        f"| Unigram | {perplexity['unigram']['perplexity']:.2f} |",
        "",
        f"Over {perplexity['transformer']['n_tokens']:,} test tokens, vocabulary "
        f"{tokenizer['vocab_size']}. ",
    ]

    if perplexity["transformer"]["perplexity"] < perplexity["bigram"]["perplexity"]:
        ratio = perplexity["bigram"]["perplexity"] / perplexity["transformer"]["perplexity"]
        lines.append(
            f"The transformer beats the bigram by {ratio:.1f}x, so it is using more than "
            "the previous event."
        )
    else:
        lines.append(
            "**The transformer does not beat the bigram.** Its surprisal still ranks "
            "moments within a match, but on this corpus it is not learning structure a "
            "count model misses."
        )

    lines += [
        "",
        f"### Downstream: {metrics['downstream']['task']}",
        "",
        "| Predictor | ROC AUC | Average precision | Brier |",
        "| --- | ---: | ---: | ---: |",
        f"| GBM on hand features | {gbm['roc_auc']:.3f} | {gbm['average_precision']:.3f} "
        f"| {gbm['brier']:.4f} |",
        f"| LM probe (frozen transformer) | {probe['roc_auc']:.3f} "
        f"| {probe['average_precision']:.3f} | {probe['brier']:.4f} |",
        f"| Base rate | {base['roc_auc']:.3f} | {base['average_precision']:.3f} "
        f"| {base['brier']:.4f} |",
        "",
        f"{gbm['n']:,} in-play events, base rate {gbm['positive_rate']:.3f}. Both "
        "predictors are fitted on the same match-disjoint split, and both are strictly "
        "causal: the probe reads the frozen model's hidden state at that event, which is "
        "computed from that event and everything before it and nothing after.",
        "",
        f"**The {winner} wins.** A hand-built feature set — field position, phase count, "
        "what has happened in the possession so far — carries more of this signal than a "
        f"linear read-out of the language model's state. The {loser} is behind it, and "
        "saying so is the point of running the comparison.",
        "",
        f"Trained in {training['seconds']:.0f}s, best validation perplexity "
        f"{training['best_val_perplexity']:.2f} at epoch {training['best_epoch']} of "
        f"{training['epochs_run']}.",
        "",
        END,
    ]
    return "\n".join(lines)


def main() -> int:
    if not METRICS.exists():
        print(f"{METRICS} is missing; run scripts/train_demo_model.py first", file=sys.stderr)
        return 1
    metrics = json.loads(METRICS.read_text())
    text = README.read_text()
    if START not in text or END not in text:
        print(f"README.md has no {START} / {END} markers", file=sys.stderr)
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    README.write_text(head + block(metrics) + tail)
    print(f"updated {README} from {METRICS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
