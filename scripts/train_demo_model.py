"""Train the model that ships with `ruck demo`, and record its metrics.

Writes into ``demo/model/``:

* ``vocab.json``     the fitted tokenizer vocabulary
* ``phase_lm.pt``    the trained checkpoint
* ``metrics.json``   perplexity against count baselines, and the downstream
                     comparison against a GBM on hand features
* ``corpus.csv.gz``  the exact synthetic corpus, so every number is reproducible

Everything here is trained on generated data (see ``ruck.demo_data.generate``).
The numbers measure the pipeline, not rugby.

    python scripts/train_demo_model.py --matches 80 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ruck.demo_data.generate import generate_corpus, write_csv
from ruck.lm import CountBaseline, LMConfig, TrainConfig, compare, evaluate, split_by_match, train
from ruck.tokenizer import Tokenizer

OUT = Path("demo/model")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20240203)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-phases", type=int, default=3, help="downstream horizon")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"generating {args.matches} synthetic matches ...")
    corpus = generate_corpus(n_matches=args.matches, seed=args.seed)
    split = split_by_match(corpus, seed=0)
    print("  " + json.dumps(split.sizes()))

    tokenizer = Tokenizer.fit(split.train, max_size=2000)
    coverage = tokenizer.coverage(split.test)
    print(f"  vocab {len(tokenizer.vocab)}  test coverage {coverage['full']:.3f} full")

    train_sequences = tokenizer.encode_events(split.train)
    val_sequences = tokenizer.encode_events(split.val)
    test_sequences = tokenizer.encode_events(split.test)

    model_config = LMConfig(vocab_size=len(tokenizer.vocab))
    print(f"training ({args.epochs} epochs max) ...")
    model, history = train(
        train_sequences,
        val_sequences,
        vocab_size=len(tokenizer.vocab),
        model_config=model_config,
        config=TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device
        ),
        checkpoint=args.out / "phase_lm.pt",
    )
    print("  " + history.summary())

    transformer = evaluate(model, test_sequences, name="transformer", device=args.device)
    unigram = (
        CountBaseline(len(tokenizer.vocab), order=1).fit(train_sequences).perplexity(test_sequences)
    )
    bigram = (
        CountBaseline(len(tokenizer.vocab), order=2).fit(train_sequences).perplexity(test_sequences)
    )
    for result in (transformer, bigram, unigram):
        print(f"  {result}")

    print(f"downstream: points within {args.n_phases} phases ...")
    downstream = compare(
        split.train, split.test, tokenizer, model, n_phases=args.n_phases, device=args.device
    )
    for result in downstream.values():
        print(f"  {result}")

    tokenizer.vocab.save(args.out / "vocab.json")
    # Gzipped: the same 80 matches as 1 MB rather than 9.
    write_csv(corpus, args.out / "corpus.csv.gz")

    metrics = {
        "generated_by": "scripts/train_demo_model.py",
        "data": {
            "synthetic": True,
            "generator": "ruck.demo_data.generate",
            "matches": args.matches,
            "seed": args.seed,
            **split.sizes(),
        },
        "tokenizer": {"vocab_size": len(tokenizer.vocab), "test_coverage": coverage},
        "model": {**asdict(model_config), "n_params": history.n_params},
        "training": {
            "epochs_run": len(history.val_perplexity),
            "best_epoch": history.best_epoch + 1,
            "best_val_perplexity": history.best_val_perplexity,
            "seconds": round(history.seconds, 1),
            "truncated_sequences": history.truncated_sequences,
            "device": str(torch.device(args.device) if args.device != "auto" else "auto"),
            "platform": platform.platform(),
        },
        "perplexity": {
            r.name: {
                "perplexity": r.perplexity,
                "nats_per_token": r.nats_per_token,
                "n_tokens": r.n_tokens,
            }
            for r in (transformer, bigram, unigram)
        },
        "downstream": {
            "task": f"points within {args.n_phases} phases",
            "results": {k: asdict(v) for k, v in downstream.items()},
        },
        "total_seconds": round(time.monotonic() - started, 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {args.out}/metrics.json in {metrics['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
