"""Tests for RetrievalDataset / RetrievalBatchSampler / RetrievalCollator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from sid_llm.training.datasets import (
    RetrievalBatchSampler,
    RetrievalCollator,
    RetrievalDataset,
)


def _write_corpus(tmp_path: Path, n_behavior: int = 6, n_metadata: int = 6) -> Path:
    rows = []
    for i in range(n_behavior):
        rows.append({
            "seq_type": "behavior",
            "input_text": f"<seq> <sid_{i}><sid_{i+1}>",
            "target_text": f"<sid_{i+2}><sid_{i+3}><sid_{i+4}><sid_{i+5}><sid_eos>",
        })
    for i in range(n_metadata):
        rows.append({
            "seq_type": "metadata",
            "input_text": f"<seq> title: item {i}",
            "target_text": f"<sid_{i*7 % 1024}><sid_{(i*11) % 1024}><sid_{(i*13) % 1024}><sid_{(i*17) % 1024}><sid_eos>",
        })
    p = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(p))
    return p


def _write_embeddings(tmp_path: Path, n_items: int = 10, dim: int = 512, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for iid in range(n_items):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        rows.append({"item_id": int(iid), "embedding": vec.tolist()})
    p = tmp_path / "embeddings.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(p))
    return p


def _write_catalog_with_sid(tmp_path: Path, n_items: int = 10, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for iid in range(n_items):
        rows.append({
            "item_id": int(iid),
            "sid_0": int(rng.integers(0, 1024)),
            "sid_1": int(rng.integers(0, 1024)),
            "sid_2": int(rng.integers(0, 1024)),
            "sid_3": int(rng.integers(0, 1024)),
        })
    p = tmp_path / "catalog_with_sid.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(p))
    return p


def test_dataset_5050_split_with_embeddings(tmp_path: Path):
    corpus = _write_corpus(tmp_path, n_behavior=4, n_metadata=4)
    emb = _write_embeddings(tmp_path, n_items=4)
    sids = _write_catalog_with_sid(tmp_path, n_items=4)
    ds = RetrievalDataset(
        corpus_path=corpus, embeddings_path=emb, catalog_with_sid_path=sids
    )
    assert ds.search_mode == "embedding"
    # min(4 behavior, 4 search items) = 4 -> 8 total rows.
    assert len(ds) == 8

    seq_count = 0
    search_count = 0
    for i in range(len(ds)):
        ex = ds[i]
        if i % 2 == 0:
            assert ex["mode"] == "sequence"
            assert ex["input_text"].startswith("<seq>")
            assert "query_embedding" not in ex
            seq_count += 1
        else:
            assert ex["mode"] == "search"
            assert isinstance(ex["query_embedding"], torch.Tensor)
            assert ex["query_embedding"].shape == (512,)
            assert ex["target_text"].endswith("<sid_eos>")
            search_count += 1
    assert seq_count == 4
    assert search_count == 4


def test_dataset_clamps_to_min_of_seq_and_search(tmp_path: Path):
    corpus = _write_corpus(tmp_path, n_behavior=10, n_metadata=2)
    emb = _write_embeddings(tmp_path, n_items=3)
    sids = _write_catalog_with_sid(tmp_path, n_items=3)
    ds = RetrievalDataset(
        corpus_path=corpus, embeddings_path=emb, catalog_with_sid_path=sids
    )
    # min(10 behavior, 3 embeddings) = 3 -> 6 total rows.
    assert len(ds) == 6


def test_dataset_smoke_cap_halves(tmp_path: Path):
    corpus = _write_corpus(tmp_path, n_behavior=20, n_metadata=20)
    emb = _write_embeddings(tmp_path, n_items=20)
    sids = _write_catalog_with_sid(tmp_path, n_items=20)
    ds = RetrievalDataset(
        corpus_path=corpus, embeddings_path=emb, catalog_with_sid_path=sids,
        smoke_cap=8,
    )
    # smoke_cap // 2 = 4 per side -> 8 total.
    assert len(ds) == 8


def test_dataset_requires_embeddings(tmp_path: Path):
    corpus = _write_corpus(tmp_path, n_behavior=2, n_metadata=2)
    with pytest.raises(ValueError, match="embeddings_path is required"):
        RetrievalDataset(corpus_path=corpus)


def test_dataset_requires_catalog_with_sid(tmp_path: Path):
    corpus = _write_corpus(tmp_path, n_behavior=2, n_metadata=2)
    emb = _write_embeddings(tmp_path, n_items=2)
    with pytest.raises(ValueError, match="catalog_with_sid_path is required"):
        RetrievalDataset(corpus_path=corpus, embeddings_path=emb)


def test_dataset_skips_items_missing_sid(tmp_path: Path):
    """Embeddings whose item_id has no SID entry are silently dropped."""
    corpus = _write_corpus(tmp_path, n_behavior=10, n_metadata=10)
    emb = _write_embeddings(tmp_path, n_items=10)
    sids = _write_catalog_with_sid(tmp_path, n_items=5)
    ds = RetrievalDataset(
        corpus_path=corpus, embeddings_path=emb, catalog_with_sid_path=sids
    )
    # n_each = min(10 behavior, 5 joined search) = 5.
    assert len(ds) == 10


def test_batch_sampler_emits_uniform_mode_batches():
    sampler = RetrievalBatchSampler(n_each=8, batch_size=4, shuffle=False, drop_last=True)
    batches = list(iter(sampler))
    assert len(batches) == len(sampler)
    assert len(batches) == 4
    for b_idx, batch in enumerate(batches):
        if b_idx % 2 == 0:
            assert all(i % 2 == 0 for i in batch), f"batch {b_idx} expected all-seq, got {batch}"
        else:
            assert all(i % 2 == 1 for i in batch), f"batch {b_idx} expected all-search, got {batch}"


def test_batch_sampler_drop_last_truncates():
    sampler = RetrievalBatchSampler(n_each=10, batch_size=4, shuffle=False, drop_last=True)
    batches = list(iter(sampler))
    # 10 // 4 = 2 batches per mode -> 4 batches total.
    assert len(batches) == 4
    for batch in batches:
        assert len(batch) == 4


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import T5TokenizerFast

    tok = T5TokenizerFast.from_pretrained("t5-small")
    extra = [f"<sid_{i}>" for i in range(64)] + [
        "<sid_pad>", "<sid_eos>", "<seq>", "<search>",
    ]
    tok.add_tokens(extra, special_tokens=True)
    return tok


def test_collator_sequence_batch(tokenizer):
    coll = RetrievalCollator(tokenizer, max_input_len=64, max_target_len=16)
    batch = [
        {"mode": "sequence", "input_text": "<seq> <sid_1>", "target_text": "<sid_2><sid_eos>"},
        {"mode": "sequence", "input_text": "<seq> <sid_3>", "target_text": "<sid_4><sid_eos>"},
    ]
    out = coll(batch)
    assert out["mode"] == "sequence"
    assert "input_ids" in out and out["input_ids"].shape[0] == 2
    assert "attention_mask" in out
    assert out["labels"].shape[0] == 2
    assert (out["labels"] == -100).any() or out["labels"].min() >= 0  # pad-masked or no pad


def test_collator_search_batch(tokenizer):
    coll = RetrievalCollator(tokenizer, max_input_len=64, max_target_len=16)
    batch = [
        {
            "mode": "search",
            "query_embedding": torch.randn(512),
            "target_text": "<sid_2><sid_eos>",
        },
        {
            "mode": "search",
            "query_embedding": torch.randn(512),
            "target_text": "<sid_4><sid_eos>",
        },
    ]
    out = coll(batch)
    assert out["mode"] == "search"
    assert out["query_embeddings"].shape == (2, 512)
    assert "input_ids" not in out
    assert out["labels"].shape[0] == 2


def test_collator_rejects_mixed_mode_batch(tokenizer):
    coll = RetrievalCollator(tokenizer)
    batch = [
        {"mode": "sequence", "input_text": "<seq> a", "target_text": "<sid_1><sid_eos>"},
        {"mode": "search", "query_embedding": torch.randn(512), "target_text": "<sid_2><sid_eos>"},
    ]
    with pytest.raises(ValueError, match="mixed modes"):
        coll(batch)
