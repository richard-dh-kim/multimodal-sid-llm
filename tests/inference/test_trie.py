import torch
from sid_llm.inference.trie import SIDTrie


def test_construction_records_valid_tuples():
    """Basic: build a trie from a few SIDs and check is_valid()."""
    sids = [
        (1, 2, 3, 4),
        (1, 2, 3, 5),
        (1, 5, 6, 7),
        (2, 3, 4, 5),
    ]
    trie = SIDTrie(sids, vocab_size=10)
    assert trie.is_valid((1, 2, 3, 4)) is True
    assert trie.is_valid((1, 2, 3, 5)) is True
    assert trie.is_valid((1, 5, 6, 7)) is True
    # Not in trie
    assert trie.is_valid((1, 2, 3, 9)) is False
    assert trie.is_valid((9, 9, 9, 9)) is False
    # Wrong length
    assert trie.is_valid((1, 2, 3)) is False
    assert trie.is_valid((1, 2, 3, 4, 5)) is False


def test_mask_logits_at_root_allows_only_first_tokens():
    """At the empty prefix, the mask should allow exactly the first-position tokens
    that appear in any valid SID."""
    sids = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
    ]
    trie = SIDTrie(sids, vocab_size=10)
    logits = torch.zeros(10)
    mask = trie.allowed_mask(prefix=())
    assert mask.shape == (10,)
    # Only tokens 1 and 5 should be allowed at the root
    assert bool(mask[1]) is True
    assert bool(mask[5]) is True
    # Everything else disallowed
    for t in range(10):
        if t not in (1, 5):
            assert bool(mask[t]) is False


def test_mask_logits_at_prefix_allows_only_extensions_in_trie():
    """Given prefix (1, 2), only tokens that are valid 3rd-position children should be allowed."""
    sids = [
        (1, 2, 3, 4),
        (1, 2, 3, 5),
        (1, 2, 7, 8),
        (1, 9, 0, 0),
    ]
    trie = SIDTrie(sids, vocab_size=10)
    mask = trie.allowed_mask(prefix=(1, 2))
    # At prefix (1,2), valid 3rd-position children are 3 and 7.
    assert bool(mask[3]) is True
    assert bool(mask[7]) is True
    for t in range(10):
        if t not in (3, 7):
            assert bool(mask[t]) is False


def test_mask_logits_at_unreachable_prefix_returns_all_disallowed():
    """A prefix that doesn't exist in the trie returns an all-False mask
    (caller's responsibility to handle, e.g., fall back to unconstrained)."""
    sids = [(1, 2, 3, 4)]
    trie = SIDTrie(sids, vocab_size=10)
    mask = trie.allowed_mask(prefix=(9, 9))
    assert mask.shape == (10,)
    assert not bool(mask.any())


def test_apply_mask_zeros_out_disallowed_logits():
    """apply_mask(prefix, logits) sets disallowed positions to a very negative value
    so a softmax assigns them ~0 probability."""
    sids = [(1, 2, 3, 4), (1, 2, 3, 5)]
    trie = SIDTrie(sids, vocab_size=10)
    logits = torch.zeros(10)
    masked = trie.apply_mask(prefix=(1, 2, 3), logits=logits)
    assert masked.shape == (10,)
    # Tokens 4 and 5 should retain their original logit (0.0)
    assert masked[4].item() == 0.0
    assert masked[5].item() == 0.0
    # Other tokens should be very negative
    for t in range(10):
        if t not in (4, 5):
            assert masked[t].item() < -1e8


def test_apply_mask_handles_batched_logits():
    """apply_mask should work on [B, V] batched logits given a single prefix."""
    sids = [(1, 2, 3, 4), (1, 2, 3, 5)]
    trie = SIDTrie(sids, vocab_size=10)
    logits = torch.zeros(3, 10)
    masked = trie.apply_mask(prefix=(1, 2, 3), logits=logits)
    assert masked.shape == (3, 10)
    # All rows should have the same mask applied
    for b in range(3):
        assert masked[b, 4].item() == 0.0
        assert masked[b, 5].item() == 0.0
        assert masked[b, 0].item() < -1e8


def test_construction_handles_large_catalog():
    """Stress: 150k random 4-token tuples with vocab_size=1024.
    Construction should finish in seconds, memory should be modest."""
    import random
    random.seed(0)
    vocab = 1024
    sids = list({tuple(random.randint(0, vocab - 1) for _ in range(4)) for _ in range(150_000)})
    # Deduplicate; expect close to 150k unique tuples since 1024^4 >> 150k
    trie = SIDTrie(sids, vocab_size=vocab)
    # Spot check a few real entries
    assert trie.is_valid(sids[0]) is True
    assert trie.is_valid(sids[-1]) is True
    # Totally random tuple unlikely to be in trie
    assert trie.is_valid((99999, 99999, 99999, 99999)) is False  # also out of vocab
