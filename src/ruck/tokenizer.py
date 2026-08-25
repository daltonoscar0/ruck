"""Deterministic event tokenizer.

A token is the four-tuple the spec asks for — action, 6x3 pitch zone,
phase-count bucket, outcome — written ``action|zone|phase|outcome``.

The full cross-product of those four fields is about 40,000 combinations, far
past the 2,000-token budget, so the vocabulary is *fitted* to a corpus under a
cap and anything outside it backs off down a cascade of progressively coarser
tokens:

    carry|opp22:M|2|retained     full
    carry|opp22:M|*|retained     drop the phase bucket
    carry|*|*|retained           drop the zone
    carry|*|*|*                  action only
    <unk>                        nothing left to say

The cascade is pure table lookup on a normalized string: no fuzzy matching, no
learned merges, and the same event always produces the same id for a given
vocabulary. Fitting adds the coarse levels first, so the action-only rung is
always present and a real event never falls all the way to ``<unk>``.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ruck.schema import ACTIONS

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")

#: Longitudinal zone edges in metres from the attacking team's own goal line,
#: chosen on rugby lines rather than equal sixths: own 22, own half, the middle
#: third split at halfway, the opposition 22, and the five-metre red zone.
X_EDGES: tuple[float, ...] = (22.0, 40.0, 60.0, 78.0, 95.0)
X_LABELS: tuple[str, ...] = ("own22", "own40", "mid", "opp40", "opp22", "opp5")

#: Lateral zone edges on the 15-metre lines: touchline channel, middle, far
#: channel. "L" and "R" are as seen by the attacking team.
Y_EDGES: tuple[float, ...] = (15.0, 55.0)
Y_LABELS: tuple[str, ...] = ("L", "M", "R")

UNKNOWN_ZONE = "??"

#: Phase-count buckets. Phase 0 is the set piece; analysts stop distinguishing
#: individual phases somewhere around six, so the tail is grouped.
PHASE_BUCKETS: tuple[tuple[int, str], ...] = (
    (0, "0"),
    (1, "1"),
    (2, "2"),
    (3, "3"),
    (5, "4-5"),
    (8, "6-8"),
)
PHASE_TAIL = "9+"
UNKNOWN_PHASE = "?"

WILDCARD = "*"
DEFAULT_MAX_VOCAB = 2000


def zone_of(x: float | None, y: float | None) -> str:
    """Map a position to one of the 18 zones, or ``??`` when unknown.

    Bins are half-open, ``[lo, hi)``: a carry taken exactly on the 22 is in
    the band above it, not below.
    """
    if x is None or y is None or (isinstance(x, float) and np.isnan(x)) or (
        isinstance(y, float) and np.isnan(y)
    ):
        return UNKNOWN_ZONE
    xi = bisect_right(X_EDGES, float(x))
    yi = bisect_right(Y_EDGES, float(y))
    return f"{X_LABELS[min(xi, len(X_LABELS) - 1)]}:{Y_LABELS[min(yi, len(Y_LABELS) - 1)]}"


def phase_bucket(phase: object) -> str:
    """Map a phase count to its bucket label."""
    if phase is None or (isinstance(phase, float) and np.isnan(phase)) or pd.isna(phase):
        return UNKNOWN_PHASE
    value = int(phase)
    for upper, label in PHASE_BUCKETS:
        if value <= upper:
            return label
    return PHASE_TAIL


@dataclass(frozen=True)
class Token:
    """The four fields of a token, before it is resolved against a vocabulary."""

    action: str
    zone: str
    phase: str
    outcome: str

    def at_level(self, level: int) -> str:
        """Render this token at a backoff level: 3 is full, 0 is action only."""
        zone = self.zone if level >= 2 else WILDCARD
        phase = self.phase if level >= 3 else WILDCARD
        outcome = self.outcome if level >= 1 else WILDCARD
        return f"{self.action}|{zone}|{phase}|{outcome}"

    def __str__(self) -> str:
        return self.at_level(3)


def token_of(row: pd.Series | dict) -> Token:
    """Build the token for one canonical event row."""
    get = row.get
    action = str(get("type") or "other")
    if action not in ACTIONS:
        action = "other"
    outcome = get("outcome")
    outcome = "unknown" if outcome is None or pd.isna(outcome) else str(outcome)
    x, y = get("x"), get("y")
    return Token(
        action=action,
        zone=zone_of(None if pd.isna(x) else float(x), None if pd.isna(y) else float(y)),
        phase=phase_bucket(get("phase")),
        outcome=outcome,
    )


class Vocab:
    """A fitted token vocabulary with a fixed size cap and a backoff cascade."""

    def __init__(self, tokens: Sequence[str]):
        if list(tokens[: len(SPECIALS)]) != list(SPECIALS):
            raise ValueError("a vocabulary must start with the special tokens")
        self.tokens: list[str] = list(tokens)
        self.index: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self.index

    @classmethod
    def fit(
        cls,
        events: pd.DataFrame | Iterable[Token],
        *,
        max_size: int = DEFAULT_MAX_VOCAB,
        backoff_share: float = 0.25,
    ) -> Vocab:
        """Fit a vocabulary under a hard size cap.

        Levels 0 (action) and 1 (action + outcome) are small and always taken
        in full, which guarantees the cascade terminates on a real token rather
        than ``<unk>``. The remaining budget is split between level 2 (the
        zone-aware backoff rung, ``backoff_share`` of it) and level 3 (the full
        token, the rest), so a diverse corpus cannot spend the whole budget on
        coarse tokens and leave the model with no phase information.
        """
        if max_size <= len(SPECIALS):
            raise ValueError(f"max_size must exceed {len(SPECIALS)}")
        if not 0.0 <= backoff_share <= 1.0:
            raise ValueError("backoff_share must be between 0 and 1")
        tokens = list(_iter_tokens(events))

        counts: list[dict[str, int]] = [{}, {}, {}, {}]
        for token in tokens:
            for level in range(4):
                key = token.at_level(level)
                counts[level][key] = counts[level].get(key, 0) + 1

        chosen: list[str] = list(SPECIALS)
        seen: set[str] = set(chosen)

        def take(level: int, budget: int) -> None:
            if budget <= 0:
                return
            # Most frequent first, ties broken by the token string, so a re-fit
            # on the same corpus is byte-identical.
            ranked = sorted(counts[level].items(), key=lambda kv: (-kv[1], kv[0]))
            added = 0
            for key, _ in ranked:
                if added >= budget:
                    break
                if key not in seen:
                    chosen.append(key)
                    seen.add(key)
                    added += 1

        take(0, max_size - len(chosen))
        take(1, max_size - len(chosen))
        remaining = max_size - len(chosen)
        take(2, int(remaining * backoff_share))
        take(3, max_size - len(chosen))
        # Anything still unspent goes back to the level-2 rung.
        take(2, max_size - len(chosen))
        return cls(chosen)

    def encode_token(self, token: Token) -> int:
        """Resolve one token through the backoff cascade."""
        for level in (3, 2, 1, 0):
            key = token.at_level(level)
            found = self.index.get(key)
            if found is not None:
                return found
        return UNK

    def backoff_level(self, token: Token) -> int:
        """Which cascade rung ``token`` resolved on; ``-1`` for ``<unk>``."""
        for level in (3, 2, 1, 0):
            if token.at_level(level) in self.index:
                return level
        return -1

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.tokens[i] if 0 <= i < len(self.tokens) else "<unk>" for i in ids]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"tokens": self.tokens}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        return cls(json.loads(Path(path).read_text())["tokens"])


def _iter_tokens(events: pd.DataFrame | Iterable[Token]) -> Iterator[Token]:
    if isinstance(events, pd.DataFrame):
        for _, row in events.iterrows():
            yield token_of(row)
    else:
        yield from events


@dataclass
class Encoded:
    """One tokenized possession, with a way back to the source events.

    ``ids`` includes ``<bos>`` and ``<eos>``; ``event_index`` is the matching
    row index in the source frame, or ``-1`` for a special token. That mapping
    is what lets the surprisal report put a timestamp on a token.
    """

    ids: np.ndarray
    event_index: np.ndarray
    match_id: str
    possession_id: int

    def __len__(self) -> int:
        return int(self.ids.size)


class Tokenizer:
    """Turns a canonical events frame into model-ready possessions."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    @classmethod
    def fit(cls, events: pd.DataFrame, *, max_size: int = DEFAULT_MAX_VOCAB) -> Tokenizer:
        return cls(Vocab.fit(events, max_size=max_size))

    def encode_events(self, events: pd.DataFrame) -> list[Encoded]:
        """Split ``events`` into possessions and encode each one."""
        if events.empty:
            return []
        out: list[Encoded] = []
        grouped = events.groupby(["match_id", "possession_id"], sort=True, dropna=False)
        for (match_id, possession_id), block in grouped:
            block = block.sort_values("t", kind="stable")
            ids = [BOS]
            index = [-1]
            for row_index, row in block.iterrows():
                ids.append(self.vocab.encode_token(token_of(row)))
                index.append(int(row_index))
            ids.append(EOS)
            index.append(-1)
            out.append(
                Encoded(
                    ids=np.asarray(ids, dtype=np.int64),
                    event_index=np.asarray(index, dtype=np.int64),
                    match_id=str(match_id),
                    possession_id=int(possession_id) if pd.notna(possession_id) else -1,
                )
            )
        return out

    def encode_stream(self, events: pd.DataFrame) -> Encoded:
        """Encode a whole match as one stream, for timeline-style analysis."""
        ids = [BOS]
        index = [-1]
        for row_index, row in events.sort_values("t", kind="stable").iterrows():
            ids.append(self.vocab.encode_token(token_of(row)))
            index.append(int(row_index))
        ids.append(EOS)
        index.append(-1)
        match_id = str(events["match_id"].iloc[0]) if len(events) else ""
        return Encoded(
            ids=np.asarray(ids, dtype=np.int64),
            event_index=np.asarray(index, dtype=np.int64),
            match_id=match_id,
            possession_id=-1,
        )

    def coverage(self, events: pd.DataFrame) -> dict[str, float]:
        """Share of events resolving at each backoff level; a corpus health check."""
        levels = [self.vocab.backoff_level(token_of(row)) for _, row in events.iterrows()]
        total = max(len(levels), 1)
        return {
            "full": levels.count(3) / total,
            "no_phase": levels.count(2) / total,
            "action_outcome": levels.count(1) / total,
            "action_only": levels.count(0) / total,
            "unk": levels.count(-1) / total,
        }
