"""Batching and splitting for the phase-play language model.

Splits are always by match, never by possession. A possession-level split would
put two possessions from the same match on both sides of the boundary, and the
model would score better for having seen that match's referee, conditions and
team pairing — the same class of leak `ruck audit` exists to find. Doing it
here would be embarrassing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ruck.tokenizer import PAD, Encoded


@dataclass
class Split:
    """A train/validation/test partition of a corpus, disjoint by match."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {
            "train_matches": self.train["match_id"].nunique(),
            "val_matches": self.val["match_id"].nunique(),
            "test_matches": self.test["match_id"].nunique(),
            "train_events": len(self.train),
            "val_events": len(self.val),
            "test_events": len(self.test),
        }


def split_by_match(
    events: pd.DataFrame,
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> Split:
    """Partition ``events`` into disjoint match sets."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    matches = np.array(sorted(events["match_id"].unique()))
    if len(matches) < 3:
        raise ValueError(f"need at least 3 matches to split, got {len(matches)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(matches)
    n_train = max(1, round(len(matches) * fractions[0]))
    n_val = max(1, round(len(matches) * fractions[1]))
    n_train = min(n_train, len(matches) - 2)
    n_val = min(n_val, len(matches) - n_train - 1)  # leave at least one test match
    groups = {
        "train": set(matches[:n_train]),
        "val": set(matches[n_train : n_train + n_val]),
        "test": set(matches[n_train + n_val :]),
    }
    return Split(
        train=events[events["match_id"].isin(groups["train"])].reset_index(drop=True),
        val=events[events["match_id"].isin(groups["val"])].reset_index(drop=True),
        test=events[events["match_id"].isin(groups["test"])].reset_index(drop=True),
    )


def pad_batch(sequences: list[Encoded], max_len: int) -> tuple[Tensor, Tensor]:
    """Right-pad a list of sequences to a common length.

    Returns ``(ids, mask)`` where ``mask`` is True on real positions. Sequences
    longer than ``max_len`` are truncated, which only bites on possessions of
    more than ~250 events; the truncation is counted by the caller.
    """
    if not sequences:
        empty = torch.zeros((0, 0), dtype=torch.long)
        return empty, empty.bool()
    width = min(max(len(s) for s in sequences), max_len)
    ids = torch.full((len(sequences), width), PAD, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        take = min(len(sequence), width)
        ids[row, :take] = torch.from_numpy(sequence.ids[:take])
        mask[row, :take] = True
    return ids, mask


def iter_batches(
    sequences: list[Encoded],
    *,
    batch_size: int,
    max_len: int,
    shuffle: bool = False,
    seed: int = 0,
):
    """Yield ``(ids, mask)`` batches, longest-first inside a batch is not needed."""
    order = np.arange(len(sequences))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [sequences[i] for i in order[start : start + batch_size]]
        yield pad_batch(chunk, max_len)


def count_truncated(sequences: list[Encoded], max_len: int) -> int:
    """How many sequences :func:`pad_batch` would cut short."""
    return sum(1 for s in sequences if len(s) > max_len)


def pick_device(preference: str = "auto") -> torch.device:
    """Choose a device. Apple silicon gets MPS, otherwise CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
