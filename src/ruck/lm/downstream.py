"""Downstream task: will this possession produce points within N phases?

The question an analyst actually asks, and the one that tells you whether the
language model learned anything a hand-built feature set does not already know.
Two predictors are fitted on the same rows and the same match-disjoint split:

* a **GBM on hand features** — field position, phase count, what has happened in
  the possession so far;
* a **linear probe on the LM's hidden state** at that event, with the
  transformer frozen.

Both are strictly causal. The LM's hidden state at position *i* is computed
from tokens 0..*i* only, and every hand feature is a running total over events
already played. The label looks strictly forward. Getting this wrong in a tool
that audits other people's leakage would be a poor look, so the boundaries are
asserted in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ruck.lm.data import pick_device
from ruck.lm.model import PhaseLM
from ruck.schema import ACTIONS, OUTCOMES
from ruck.tokenizer import Tokenizer

#: Only in-play moments are scored. Asking "will they score in three phases?"
#: at a substitution is not a question anyone has.
IN_PLAY: frozenset[str] = frozenset(
    {"carry", "pass", "offload", "ruck", "maul", "lineout", "scrum", "kick", "restart", "tackle"}
)

HAND_FEATURES: tuple[str, ...] = (
    "phase",
    "x",
    "y",
    "dist_to_line",
    "lateral_offset",
    "seconds_in_possession",
    "events_in_possession",
    "passes_so_far",
    "rucks_so_far",
    "carries_so_far",
    "metres_last_three",
    "action_id",
    "outcome_id",
    "period",
    "start_kind_id",
)

_ACTION_ID = {a: i for i, a in enumerate(ACTIONS)}
_OUTCOME_ID = {o: i for i, o in enumerate(OUTCOMES)}
_START_KIND_ID = {"lineout": 0, "scrum": 1, "restart": 2, "ruck": 3, "penalty": 4}


@dataclass
class DownstreamResult:
    """Scores for one predictor on one split."""

    name: str
    roc_auc: float
    average_precision: float
    brier: float
    n: int
    positive_rate: float

    def __str__(self) -> str:
        return (
            f"{self.name}: AUC {self.roc_auc:.3f}  AP {self.average_precision:.3f}  "
            f"Brier {self.brier:.4f}  (n={self.n:,}, base rate {self.positive_rate:.3f})"
        )


def build_labels(events: pd.DataFrame, n_phases: int = 3) -> pd.Series:
    """Label each event with whether its possession scores within ``n_phases``.

    Strictly forward-looking: only events later in the same possession count,
    and only if they land on or before ``phase + n_phases``.
    """
    labels = np.zeros(len(events), dtype=np.int8)
    for _, block in events.groupby(["match_id", "possession_id"], sort=False):
        positions = block.index.to_numpy()
        phases = block["phase"].fillna(0).astype(int).to_numpy()
        scoring = (block["outcome"] == "points").to_numpy()
        for offset in range(len(block)):
            horizon = phases[offset] + n_phases
            future = np.arange(offset + 1, len(block))
            if future.size == 0:
                continue
            hits = scoring[future] & (phases[future] <= horizon)
            if bool(hits.any()):
                labels[events.index.get_loc(positions[offset])] = 1
    return pd.Series(labels, index=events.index, name=f"points_within_{n_phases}")


def build_hand_features(events: pd.DataFrame) -> pd.DataFrame:
    """Running, causal features for each event."""
    frames = []
    for _, block in events.groupby(["match_id", "possession_id"], sort=False):
        block = block.sort_values("t", kind="stable")
        x = block["x"].to_numpy(dtype=float)
        start_t = float(block["t"].iloc[0])
        actions = block["type"].astype(str).to_numpy()

        passes = np.cumsum(np.isin(actions, ["pass", "offload"])) - np.isin(
            actions, ["pass", "offload"]
        )
        rucks = np.cumsum(np.isin(actions, ["ruck", "maul"])) - np.isin(actions, ["ruck", "maul"])
        carries = np.cumsum(actions == "carry") - (actions == "carry")

        lagged = (
            np.concatenate([np.full(min(3, len(x)), np.nan), x[:-3]])
            if len(x) > 3
            else (np.full(len(x), np.nan))
        )
        start_kind = actions[0] if len(actions) else "other"

        frames.append(
            pd.DataFrame(
                {
                    "phase": block["phase"].fillna(0).astype(float).to_numpy(),
                    "x": x,
                    "y": block["y"].to_numpy(dtype=float),
                    "dist_to_line": 100.0 - x,
                    "lateral_offset": np.abs(block["y"].to_numpy(dtype=float) - 35.0),
                    "seconds_in_possession": block["t"].to_numpy(dtype=float) - start_t,
                    "events_in_possession": np.arange(len(block), dtype=float),
                    "passes_so_far": passes.astype(float),
                    "rucks_so_far": rucks.astype(float),
                    "carries_so_far": carries.astype(float),
                    "metres_last_three": x - lagged,
                    "action_id": [float(_ACTION_ID.get(a, len(ACTIONS))) for a in actions],
                    "outcome_id": [
                        float(_OUTCOME_ID.get(str(o), len(OUTCOMES)))
                        for o in block["outcome"].astype(str)
                    ],
                    "period": block["period"].fillna(1).astype(float).to_numpy(),
                    "start_kind_id": float(_START_KIND_ID.get(start_kind, len(_START_KIND_ID))),
                },
                index=block.index,
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(HAND_FEATURES))
    return pd.concat(frames).reindex(events.index)[list(HAND_FEATURES)]


@torch.no_grad()
def lm_states(
    events: pd.DataFrame,
    tokenizer: Tokenizer,
    model: PhaseLM,
    *,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """The frozen LM's hidden state at each event, indexed like ``events``.

    The state at an event's position is a function of that event and everything
    before it, and of nothing after it: the model is causal and is run over the
    possession in order.
    """
    model.eval()
    model.to(device)
    width = model.config.d_model
    out = np.full((len(events), width), np.nan, dtype=np.float32)
    row_of = {index: position for position, index in enumerate(events.index)}

    for sequence in tokenizer.encode_events(events):
        ids = torch.from_numpy(sequence.ids[: model.config.max_len]).unsqueeze(0).to(device)
        _, hidden = model(ids)
        hidden = hidden.squeeze(0).cpu().numpy()
        for position, event_index in enumerate(sequence.event_index[: model.config.max_len]):
            if event_index < 0:
                continue
            row = row_of.get(event_index)
            if row is not None:
                out[row] = hidden[position]
    return pd.DataFrame(out, index=events.index)


def _score(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> DownstreamResult:
    positive_rate = float(y_true.mean()) if y_true.size else float("nan")
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return DownstreamResult(
            name, float("nan"), float("nan"), float("nan"), int(y_true.size), positive_rate
        )
    return DownstreamResult(
        name=name,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        average_precision=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        n=int(y_true.size),
        positive_rate=positive_rate,
    )


def compare(
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    tokenizer: Tokenizer,
    model: PhaseLM,
    *,
    n_phases: int = 3,
    device: str = "auto",
    seed: int = 0,
) -> dict[str, DownstreamResult]:
    """Fit both predictors on ``train_events`` and score them on ``test_events``."""
    torch_device = pick_device(device)

    def prepare(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        labels = build_labels(events, n_phases=n_phases)
        hand = build_hand_features(events)
        states = lm_states(events, tokenizer, model, device=torch_device)
        keep = events["type"].isin(IN_PLAY).to_numpy() & states.notna().all(axis=1).to_numpy()
        return hand[keep], states[keep], labels[keep].to_numpy()

    hand_train, states_train, y_train = prepare(train_events)
    hand_test, states_test, y_test = prepare(test_events)

    results: dict[str, DownstreamResult] = {}

    base_rate = float(y_train.mean()) if y_train.size else 0.0
    results["base_rate"] = _score(
        "base rate", y_test, np.full(y_test.shape, base_rate, dtype=float)
    )

    gbm = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=None,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    gbm.fit(hand_train.to_numpy(), y_train)
    results["gbm"] = _score(
        "GBM on hand features", y_test, gbm.predict_proba(hand_test.to_numpy())[:, 1]
    )

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, random_state=seed),
    )
    probe.fit(states_train.to_numpy(), y_train)
    results["lm_probe"] = _score(
        "LM probe (frozen)", y_test, probe.predict_proba(states_test.to_numpy())[:, 1]
    )

    return results
