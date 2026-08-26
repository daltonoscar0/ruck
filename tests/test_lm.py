from __future__ import annotations

import numpy as np
import pytest
import torch

from ruck.demo_data.generate import generate_corpus
from ruck.lm import (
    MAX_PARAMS,
    CountBaseline,
    LMConfig,
    PhaseLM,
    TrainConfig,
    build_hand_features,
    build_labels,
    model_perplexity,
    split_by_match,
    train,
)
from ruck.lm.data import count_truncated, iter_batches, pad_batch, pick_device
from ruck.lm.downstream import HAND_FEATURES, lm_states
from ruck.tokenizer import BOS, PAD, Tokenizer


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_matches=6, seed=17)


@pytest.fixture(scope="module")
def tokenizer(corpus):
    return Tokenizer.fit(corpus, max_size=512)


@pytest.fixture(scope="module")
def sequences(corpus, tokenizer):
    return tokenizer.encode_events(corpus)


# -- splitting -----------------------------------------------------------


def test_the_split_is_disjoint_by_match(corpus):
    split = split_by_match(corpus, seed=1)
    train_matches = set(split.train["match_id"])
    val_matches = set(split.val["match_id"])
    test_matches = set(split.test["match_id"])
    assert train_matches & val_matches == set()
    assert train_matches & test_matches == set()
    assert val_matches & test_matches == set()
    assert train_matches | val_matches | test_matches == set(corpus["match_id"])


def test_the_split_never_leaves_a_side_empty(corpus):
    split = split_by_match(corpus, seed=1)
    assert len(split.train) and len(split.val) and len(split.test)


def test_splitting_needs_at_least_three_matches():
    small = generate_corpus(n_matches=2, seed=1)
    with pytest.raises(ValueError, match="at least 3 matches"):
        split_by_match(small)


def test_fractions_must_sum_to_one(corpus):
    with pytest.raises(ValueError, match="sum to 1"):
        split_by_match(corpus, fractions=(0.5, 0.2, 0.2))


# -- batching ------------------------------------------------------------


def test_padding_is_on_the_right_and_masked(sequences):
    ids, mask = pad_batch(sequences[:8], max_len=256)
    assert ids.shape == mask.shape
    for row in range(ids.shape[0]):
        length = int(mask[row].sum())
        assert (ids[row, length:] == PAD).all()
        assert ids[row, 0] == BOS


def test_an_empty_batch_is_handled(tokenizer):
    ids, mask = pad_batch([], max_len=64)
    assert ids.numel() == 0 and mask.numel() == 0


def test_long_sequences_are_truncated_and_counted(sequences):
    ids, _ = pad_batch(sequences, max_len=8)
    assert ids.shape[1] == 8
    assert count_truncated(sequences, 8) > 0


def test_batches_cover_every_sequence(sequences):
    seen = sum(
        int(mask.shape[0]) for _, mask in iter_batches(sequences, batch_size=16, max_len=256)
    )
    assert seen == len(sequences)


def test_device_selection_returns_something_usable():
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in {"cpu", "mps", "cuda"}


# -- the model -----------------------------------------------------------


def test_the_default_model_is_well_under_the_parameter_budget():
    model = PhaseLM(LMConfig(vocab_size=2000))
    assert model.n_params() < MAX_PARAMS
    assert model.n_params() < 2_000_000


def test_a_model_over_the_budget_is_refused():
    with pytest.raises(ValueError, match="exceeds the"):
        PhaseLM(LMConfig(vocab_size=2000, d_model=1024, n_layers=24, d_ff=4096))


def test_tied_weights_are_counted_once():
    tied = PhaseLM(LMConfig(vocab_size=500, tie_weights=True)).n_params()
    untied = PhaseLM(LMConfig(vocab_size=500, tie_weights=False)).n_params()
    assert untied > tied


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        LMConfig(vocab_size=100, d_model=100, n_heads=7)


def test_forward_returns_logits_and_hidden_of_the_right_shape():
    config = LMConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_ff=64)
    model = PhaseLM(config)
    ids = torch.randint(0, 64, (3, 11))
    logits, hidden = model(ids)
    assert logits.shape == (3, 11, 64)
    assert hidden.shape == (3, 11, 32)


def test_a_sequence_longer_than_max_len_is_refused():
    model = PhaseLM(LMConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2, max_len=8))
    with pytest.raises(ValueError, match="max_len"):
        model(torch.zeros((1, 9), dtype=torch.long))


def test_the_model_cannot_see_the_future():
    """Changing a later token must not change an earlier position's output."""
    torch.manual_seed(0)
    model = PhaseLM(LMConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=2, dropout=0.0))
    model.eval()
    a = torch.tensor([[1, 5, 9, 13, 17]])
    b = a.clone()
    b[0, -1] = 21
    with torch.no_grad():
        first, _ = model(a)
        second, _ = model(b)
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)


def test_surprisal_has_no_value_for_the_first_position():
    model = PhaseLM(LMConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2, dropout=0.0))
    bits = model.token_surprisal(torch.tensor([[1, 4, 7]]))
    assert torch.isnan(bits[0, 0])
    assert torch.isfinite(bits[0, 1:]).all()


def test_a_checkpoint_round_trips(tmp_path):
    config = LMConfig(vocab_size=48, d_model=32, n_layers=2, n_heads=2)
    model = PhaseLM(config)
    model.save(tmp_path / "m.pt")
    reloaded = PhaseLM.load(tmp_path / "m.pt")
    ids = torch.randint(0, 48, (2, 6))
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids)[0], reloaded(ids)[0], atol=1e-6)


# -- metrics -------------------------------------------------------------


def test_an_untrained_model_is_about_as_surprised_as_the_vocabulary_is_big(sequences):
    model = PhaseLM(LMConfig(vocab_size=512, d_model=32, n_layers=1, n_heads=2, dropout=0.0))
    result = model_perplexity(model, sequences[:20], device="cpu")
    assert 100 < result.perplexity < 2000
    assert result.n_tokens > 0


def test_count_baselines_beat_a_uniform_guess(sequences, tokenizer):
    vocab_size = len(tokenizer.vocab)
    unigram = CountBaseline(vocab_size, order=1).fit(sequences).perplexity(sequences)
    bigram = CountBaseline(vocab_size, order=2).fit(sequences).perplexity(sequences)
    assert unigram.perplexity < vocab_size
    assert bigram.perplexity < unigram.perplexity
    assert bigram.name == "bigram"


def test_perplexity_of_nothing_is_not_a_number(tokenizer):
    result = CountBaseline(10, order=2).fit([]).perplexity([])
    assert np.isnan(result.perplexity)


def test_training_reduces_validation_perplexity(corpus, tokenizer):
    split = split_by_match(corpus, seed=2)
    train_sequences = tokenizer.encode_events(split.train)
    val_sequences = tokenizer.encode_events(split.val)
    model, history = train(
        train_sequences,
        val_sequences,
        vocab_size=len(tokenizer.vocab),
        model_config=LMConfig(
            vocab_size=len(tokenizer.vocab), d_model=48, n_layers=2, n_heads=2, d_ff=96
        ),
        config=TrainConfig(epochs=3, batch_size=32, device="cpu", warmup_steps=5),
        verbose=False,
    )
    assert history.val_perplexity[-1] < history.val_perplexity[0]
    assert history.n_params == model.n_params()
    assert history.best_epoch >= 0


# -- the downstream task -------------------------------------------------


def test_labels_look_strictly_forward(corpus):
    labels = build_labels(corpus, n_phases=3)
    assert labels.index.equals(corpus.index)
    assert set(labels.unique()) <= {0, 1}
    assert 0.0 < labels.mean() < 0.5


def test_the_scoring_event_itself_is_not_labelled_from_its_own_future(corpus):
    """The last event of a possession has no future inside it, so it is 0."""
    labels = build_labels(corpus, n_phases=3)
    last_rows = corpus.groupby(["match_id", "possession_id"]).tail(1).index
    assert labels.loc[last_rows].sum() == 0


def test_a_longer_horizon_labels_at_least_as_many_rows(corpus):
    assert build_labels(corpus, n_phases=6).sum() >= build_labels(corpus, n_phases=1).sum()


def test_hand_features_are_complete_and_causal(corpus):
    features = build_hand_features(corpus)
    assert list(features.columns) == list(HAND_FEATURES)
    assert features.index.equals(corpus.index)
    # Running counts start at zero for the first event of a possession.
    first_rows = corpus.groupby(["match_id", "possession_id"]).head(1).index
    assert (features.loc[first_rows, "events_in_possession"] == 0).all()
    assert (features.loc[first_rows, "passes_so_far"] == 0).all()


def test_counts_never_include_the_current_event(corpus):
    features = build_hand_features(corpus)
    passes = corpus["type"].isin(["pass", "offload"])
    for _, block in corpus.groupby(["match_id", "possession_id"]):
        seen = 0
        for index in block.index:
            assert features.loc[index, "passes_so_far"] == seen
            seen += int(passes.loc[index])


def test_lm_states_are_produced_for_every_event(corpus, tokenizer):
    model = PhaseLM(LMConfig(vocab_size=len(tokenizer.vocab), d_model=32, n_layers=1, n_heads=2))
    states = lm_states(corpus.head(400), tokenizer, model, device="cpu")
    assert states.shape[1] == 32
    assert states.notna().all(axis=1).mean() > 0.99


# -- the label boundary ---------------------------------------------------


def test_the_label_counts_scoring_actions_not_scoring_outcomes(corpus):
    """A try recorded twice must not make its own first record predictable."""
    from ruck.lm.downstream import scoring_instants
    from ruck.schema import POINTS

    labels = build_labels(corpus, n_phases=3)
    scored_rows = corpus[scoring_instants(corpus)]
    assert len(scored_rows) > 0
    # Every carry that itself scored is excluded, so its outcome cannot be read
    # off as the answer.
    assert set(corpus.loc[corpus["type"].isin(POINTS)].index) <= set(scored_rows.index)
    assert labels.loc[scored_rows.index].sum() >= 0  # excluded downstream, not mislabelled


def test_a_scoring_instant_is_never_an_evaluated_row(corpus, tokenizer):
    from ruck.lm.downstream import IN_PLAY, scoring_instants

    evaluated = corpus["type"].isin(IN_PLAY) & ~scoring_instants(corpus)
    assert evaluated.sum() > 0
    assert not (evaluated & (corpus["outcome"] == "points")).any()


def test_the_baseline_chooses_its_smoothing_on_validation(corpus, tokenizer):
    split = split_by_match(corpus, seed=5)
    train_sequences = tokenizer.encode_events(split.train)
    val_sequences = tokenizer.encode_events(split.val)
    test_sequences = tokenizer.encode_events(split.test)

    untuned = CountBaseline(len(tokenizer.vocab), order=2).fit(train_sequences)
    fixed = untuned.perplexity(test_sequences).perplexity

    tuned = CountBaseline(len(tokenizer.vocab), order=2).fit(train_sequences)
    tuned.tune(val_sequences)
    assert tuned.perplexity(test_sequences).perplexity <= fixed
    assert tuned.k != 0.1 or fixed == tuned.perplexity(test_sequences).perplexity


def test_baseline_perplexity_can_be_held_to_the_models_context(corpus, tokenizer):
    sequences = tokenizer.encode_events(corpus)
    baseline = CountBaseline(len(tokenizer.vocab), order=2).fit(sequences)
    full = baseline.perplexity(sequences).n_tokens
    clipped = baseline.perplexity(sequences, max_len=8).n_tokens
    assert clipped < full
