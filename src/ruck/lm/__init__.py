"""The phase-play language model and the metrics that keep it honest."""

from ruck.lm.data import Split, pick_device, split_by_match
from ruck.lm.downstream import DownstreamResult, build_hand_features, build_labels, compare
from ruck.lm.metrics import CountBaseline, PerplexityResult, model_perplexity, sequence_surprisal
from ruck.lm.model import MAX_PARAMS, LMConfig, PhaseLM
from ruck.lm.train import TrainConfig, TrainHistory, evaluate, train

__all__ = [
    "MAX_PARAMS",
    "CountBaseline",
    "DownstreamResult",
    "LMConfig",
    "PerplexityResult",
    "PhaseLM",
    "Split",
    "TrainConfig",
    "TrainHistory",
    "build_hand_features",
    "build_labels",
    "compare",
    "evaluate",
    "model_perplexity",
    "pick_device",
    "sequence_surprisal",
    "split_by_match",
    "train",
]
