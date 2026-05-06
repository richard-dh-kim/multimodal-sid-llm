"""Smoke tests for BeamSearchRetriever against the M3.5 init checkpoint."""
from pathlib import Path

import pytest

from sid_llm.inference.beam_search import load_retriever


INIT_DIR = Path("checkpoints/sid_llm/init/hf_model")
SID_TO_ITEM = Path("data/catalog/sid_to_item.pkl")
SID_TRIE = Path("data/catalog/sid_trie.pkl")


@pytest.mark.skipif(
    not INIT_DIR.exists() or not SID_TO_ITEM.exists() or not SID_TRIE.exists(),
    reason="Requires M3.4 + M3.5 artifacts on disk; skipped in CI.",
)
def test_retriever_returns_k_results_constrained():
    retr = load_retriever(INIT_DIR, SID_TO_ITEM, SID_TRIE, device="cpu")
    item_ids, sids = retr.retrieve_from_text(
        "cordless drill with two batteries", k=5, num_beams=8, constrained=True
    )
    assert len(item_ids) == 5
    assert len(sids) == 5
    # With constrained decoding, all SIDs should be in sid_to_item (no silent misses).
    for sid in sids:
        assert sid in retr.sid_to_item, f"constrained decoding emitted invalid SID {sid}"


@pytest.mark.skipif(
    not INIT_DIR.exists() or not SID_TO_ITEM.exists() or not SID_TRIE.exists(),
    reason="Requires M3.4 + M3.5 artifacts on disk; skipped in CI.",
)
def test_retriever_unconstrained_may_emit_invalid_sids():
    """Without constrained decoding, the untrained init model can emit SIDs not in the catalog."""
    retr = load_retriever(INIT_DIR, SID_TO_ITEM, SID_TRIE, device="cpu")
    item_ids, sids = retr.retrieve_from_text(
        "cordless drill with two batteries", k=5, num_beams=8, constrained=False
    )
    # Just check shape; we don't assert hallucination rate (untrained model is unpredictable).
    assert len(item_ids) == 5
