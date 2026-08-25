from __future__ import annotations

import pandas as pd
import pytest

from ruck.demo_data.generate import generate_corpus, generate_match
from ruck.schema import coerce
from ruck.tokenizer import (
    BOS,
    DEFAULT_MAX_VOCAB,
    EOS,
    SPECIALS,
    UNK,
    Token,
    Tokenizer,
    Vocab,
    phase_bucket,
    token_of,
    zone_of,
)


@pytest.mark.parametrize(
    ("x", "y", "zone"),
    [
        (0.0, 0.0, "own22:L"),
        (21.9, 35.0, "own22:M"),
        (22.0, 35.0, "own40:M"),  # bins are half-open, [lo, hi)
        (22.1, 35.0, "own40:M"),
        (50.0, 14.9, "mid:L"),
        (50.0, 15.0, "mid:M"),
        (50.0, 15.1, "mid:M"),
        (50.0, 55.1, "mid:R"),
        (96.0, 69.0, "opp5:R"),
        (100.0, 70.0, "opp5:R"),
    ],
)
def test_zone_boundaries(x, y, zone):
    assert zone_of(x, y) == zone


def test_zone_is_unknown_without_coordinates():
    assert zone_of(None, None) == "??"
    assert zone_of(50.0, None) == "??"


def test_there_are_eighteen_zones():
    zones = {zone_of(x, y) for x in range(0, 101, 1) for y in range(0, 71, 1)}
    assert len(zones) == 18


@pytest.mark.parametrize(
    ("phase", "bucket"),
    [(0, "0"), (1, "1"), (3, "3"), (4, "4-5"), (5, "4-5"), (6, "6-8"), (8, "6-8"), (9, "9+"),
     (25, "9+"), (None, "?")],
)
def test_phase_buckets(phase, bucket):
    assert phase_bucket(phase) == bucket


def test_token_backoff_levels_render_as_expected():
    token = Token(action="carry", zone="opp22:M", phase="2", outcome="retained")
    assert token.at_level(3) == "carry|opp22:M|2|retained"
    assert token.at_level(2) == "carry|opp22:M|*|retained"
    assert token.at_level(1) == "carry|*|*|retained"
    assert token.at_level(0) == "carry|*|*|*"


def test_token_of_reads_a_canonical_row():
    frame = coerce(
        pd.DataFrame([{"match_id": "m", "t": 0.0, "type": "carry", "x": 80.0, "y": 35.0,
                       "phase": 2, "outcome": "retained"}])
    )
    assert str(token_of(frame.iloc[0])) == "carry|opp22:M|2|retained"


def test_an_action_outside_the_vocabulary_becomes_other():
    frame = coerce(pd.DataFrame([{"match_id": "m", "t": 0.0, "type": "carry"}]))
    row = frame.iloc[0].copy()
    row["type"] = "scrummage"
    assert token_of(row).action == "other"


# -- vocabulary ----------------------------------------------------------


def test_vocab_stays_under_the_budget():
    corpus = generate_corpus(n_matches=8, seed=11)
    vocab = Vocab.fit(corpus, max_size=DEFAULT_MAX_VOCAB)
    assert len(vocab) <= DEFAULT_MAX_VOCAB
    assert len(vocab) < 2000


def test_vocab_starts_with_the_special_tokens():
    vocab = Vocab.fit(generate_match(seed=1, home="Kestrels", away="Ravens"))
    assert vocab.tokens[: len(SPECIALS)] == list(SPECIALS)
    assert vocab.index["<bos>"] == BOS and vocab.index["<eos>"] == EOS


def test_fitting_twice_on_the_same_corpus_is_byte_identical():
    corpus = generate_corpus(n_matches=4, seed=3)
    assert Vocab.fit(corpus).tokens == Vocab.fit(corpus).tokens


def test_a_tight_budget_still_covers_every_event_by_backing_off():
    corpus = generate_corpus(n_matches=6, seed=5)
    tight = Tokenizer(Vocab.fit(corpus, max_size=64))
    coverage = tight.coverage(corpus)
    assert coverage["unk"] == 0.0
    # A 64-token budget cannot hold the full cross-product, so most events must
    # have resolved on a coarser rung.
    assert coverage["full"] < 0.5


def test_a_generous_budget_resolves_almost_everything_at_full_detail():
    corpus = generate_corpus(n_matches=6, seed=5)
    tokenizer = Tokenizer.fit(corpus, max_size=DEFAULT_MAX_VOCAB)
    assert tokenizer.coverage(corpus)["full"] > 0.95


def test_an_unseen_token_backs_off_rather_than_hitting_unk():
    corpus = generate_corpus(n_matches=3, seed=9)
    vocab = Vocab.fit(corpus)
    unseen = Token(action="carry", zone="opp5:L", phase="9+", outcome="held")
    assert vocab.encode_token(unseen) != UNK
    assert vocab.backoff_level(unseen) < 3


def test_an_action_never_seen_at_all_falls_to_unk():
    frame = coerce(pd.DataFrame([{"match_id": "m", "t": 0.0, "type": "carry"}]))
    vocab = Vocab.fit(frame)
    assert vocab.encode_token(Token("mark", "mid:M", "0", "won")) == UNK


def test_vocab_round_trips_through_disk(tmp_path):
    vocab = Vocab.fit(generate_corpus(n_matches=2, seed=2))
    path = tmp_path / "vocab.json"
    vocab.save(path)
    assert Vocab.load(path).tokens == vocab.tokens


def test_vocab_rejects_a_token_list_without_specials():
    with pytest.raises(ValueError, match="special tokens"):
        Vocab(["carry|*|*|*"])


def test_fit_rejects_an_impossible_budget():
    with pytest.raises(ValueError, match="max_size"):
        Vocab.fit(generate_match(seed=1, home="Kestrels", away="Ravens"), max_size=3)


# -- encoding ------------------------------------------------------------


def test_possessions_are_wrapped_in_bos_and_eos():
    match = generate_match(seed=4, home="Otters", away="Stags")
    tokenizer = Tokenizer.fit(match)
    encoded = tokenizer.encode_events(match)
    assert encoded
    for sequence in encoded:
        assert sequence.ids[0] == BOS and sequence.ids[-1] == EOS
        assert sequence.event_index[0] == -1 and sequence.event_index[-1] == -1


def test_event_index_points_back_at_the_source_rows():
    match = generate_match(seed=4, home="Otters", away="Stags")
    tokenizer = Tokenizer.fit(match)
    for sequence in tokenizer.encode_events(match):
        real = sequence.event_index[sequence.event_index >= 0]
        assert match.loc[real, "possession_id"].nunique() == 1


def test_every_event_is_encoded_exactly_once_across_possessions():
    match = generate_match(seed=6, home="Herons", away="Badgers")
    tokenizer = Tokenizer.fit(match)
    indices = [i for s in tokenizer.encode_events(match) for i in s.event_index if i >= 0]
    assert sorted(indices) == list(range(len(match)))


def test_encode_stream_covers_the_whole_match_in_order():
    match = generate_match(seed=8, home="Ravens", away="Kestrels")
    tokenizer = Tokenizer.fit(match)
    stream = tokenizer.encode_stream(match)
    assert len(stream) == len(match) + 2
    real = stream.event_index[stream.event_index >= 0]
    assert list(real) == list(range(len(match)))


def test_encoding_is_stable_across_calls():
    match = generate_match(seed=10, home="Stags", away="Herons")
    tokenizer = Tokenizer.fit(match)
    first = [s.ids.tolist() for s in tokenizer.encode_events(match)]
    second = [s.ids.tolist() for s in tokenizer.encode_events(match)]
    assert first == second


def test_encoding_an_empty_frame_returns_nothing():
    tokenizer = Tokenizer.fit(generate_match(seed=1, home="Otters", away="Stags"))
    assert tokenizer.encode_events(coerce(pd.DataFrame())) == []


def test_decode_returns_the_token_strings():
    match = generate_match(seed=12, home="Badgers", away="Otters")
    tokenizer = Tokenizer.fit(match)
    sequence = tokenizer.encode_events(match)[0]
    decoded = tokenizer.vocab.decode(sequence.ids.tolist())
    assert decoded[0] == "<bos>" and decoded[-1] == "<eos>"
    assert all("|" in token for token in decoded[1:-1])
