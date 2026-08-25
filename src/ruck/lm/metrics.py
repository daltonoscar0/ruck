"""Perplexity, and the baselines that make it mean something.

A perplexity number on its own says nothing — the vocabulary size and the token
design set the scale. So the model is always reported next to two count-based
baselines fitted on the same training split: a unigram model and an
add-k-smoothed bigram model. If the transformer cannot beat the bigram, the
report says so.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch
from torch.nn import functional as F

from ruck.lm.data import iter_batches
from ruck.lm.model import PhaseLM
from ruck.tokenizer import PAD, Encoded


@dataclass
class PerplexityResult:
    """Cross-entropy and perplexity over a set of sequences."""

    name: str
    nats_per_token: float
    perplexity: float
    n_tokens: int

    def __str__(self) -> str:
        return f"{self.name}: ppl {self.perplexity:.3f} over {self.n_tokens:,} tokens"


@torch.no_grad()
def model_perplexity(
    model: PhaseLM,
    sequences: list[Encoded],
    *,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    name: str = "transformer",
) -> PerplexityResult:
    """Next-event perplexity, averaged over every predicted position."""
    model.eval()
    model.to(device)
    total_nats = 0.0
    total_tokens = 0
    for ids, mask in iter_batches(sequences, batch_size=batch_size, max_len=model.config.max_len):
        if ids.numel() == 0:
            continue
        ids = ids.to(device)
        mask = mask.to(device)
        logits, _ = model(ids)
        # Predict position i+1 from everything up to i.
        targets = ids[:, 1:]
        valid = mask[:, 1:]
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)
        total_nats += float((losses * valid).sum().item())
        total_tokens += int(valid.sum().item())
    if total_tokens == 0:
        return PerplexityResult(name, float("nan"), float("nan"), 0)
    nats = total_nats / total_tokens
    return PerplexityResult(name, nats, math.exp(nats), total_tokens)


class CountBaseline:
    """Add-k smoothed unigram or bigram model over token ids."""

    def __init__(self, vocab_size: int, order: int = 2, k: float = 0.1):
        if order not in (1, 2):
            raise ValueError("order must be 1 (unigram) or 2 (bigram)")
        self.vocab_size = vocab_size
        self.order = order
        self.k = k
        self.unigram: Counter[int] = Counter()
        self.bigram: dict[int, Counter[int]] = {}
        self.total = 0

    def fit(self, sequences: list[Encoded]) -> CountBaseline:
        for sequence in sequences:
            ids = sequence.ids
            for token in ids:
                self.unigram[int(token)] += 1
                self.total += 1
            if self.order == 2:
                for previous, current in pairwise(ids):
                    self.bigram.setdefault(int(previous), Counter())[int(current)] += 1
        return self

    def log_prob(self, previous: int | None, current: int) -> float:
        denominator_extra = self.k * self.vocab_size
        if self.order == 2 and previous is not None:
            row = self.bigram.get(previous)
            if row is not None:
                return math.log((row[current] + self.k) / (sum(row.values()) + denominator_extra))
        return math.log((self.unigram[current] + self.k) / (self.total + denominator_extra))

    def perplexity(self, sequences: list[Encoded], name: str | None = None) -> PerplexityResult:
        total_nats = 0.0
        total_tokens = 0
        for sequence in sequences:
            ids = [int(i) for i in sequence.ids]
            for position in range(1, len(ids)):
                if ids[position] == PAD:
                    continue
                total_nats -= self.log_prob(ids[position - 1], ids[position])
                total_tokens += 1
        label = name or ("bigram" if self.order == 2 else "unigram")
        if total_tokens == 0:
            return PerplexityResult(label, float("nan"), float("nan"), 0)
        nats = total_nats / total_tokens
        return PerplexityResult(label, nats, math.exp(nats), total_tokens)


@torch.no_grad()
def sequence_surprisal(
    model: PhaseLM, sequence: Encoded, *, device: torch.device | str = "cpu"
) -> np.ndarray:
    """Per-position surprisal in bits for one sequence, aligned to ``sequence.ids``."""
    model.eval()
    model.to(device)
    ids = torch.from_numpy(sequence.ids[: model.config.max_len]).unsqueeze(0).to(device)
    bits = model.token_surprisal(ids).squeeze(0).cpu().numpy()
    if len(sequence) > model.config.max_len:
        tail = np.full(len(sequence) - model.config.max_len, np.nan)
        bits = np.concatenate([bits, tail])
    return bits
