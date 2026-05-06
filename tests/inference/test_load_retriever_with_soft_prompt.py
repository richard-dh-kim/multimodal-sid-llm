"""CPU-only test: `load_retriever` discovers and loads `soft_prompt.pt`.

We simulate an M3.7 checkpoint directory by saving a tiny T5 + tokenizer plus
a hand-crafted `soft_prompt.pt` (matching the on-disk layout written by
`train_retrieval.HFSavePerEpoch`). Then we call `load_retriever` and verify
that:
  1. The returned retriever has `query_projection` and `soft_prompt_offsets`.
  2. The loaded weights match what we wrote to disk.
  3. The retriever can actually call `retrieve_from_query_embedding` end-to-end.
  4. A checkpoint without `soft_prompt.pt` returns a sequence-mode-only
     retriever (no crash on load, but search mode raises).
"""
from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast

from sid_llm.inference.beam_search import load_retriever
from sid_llm.inference.trie import SIDTrie


D_MODEL = 64
QUERY_DIM = 512
NUM_SOFT = 4
NUM_SIDS = 16


def _make_tiny_t5_dir(out_dir: Path) -> None:
    cfg = T5Config(
        vocab_size=32128,
        d_model=D_MODEL,
        d_kv=32,
        d_ff=128,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        relative_attention_num_buckets=8,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(cfg)
    tok = T5TokenizerFast.from_pretrained("t5-small")
    extras = (
        [f"<sid_{i}>" for i in range(1024)]
        + ["<sid_pad>", "<sid_eos>", "<seq>", "<search>"]
    )
    tok.add_tokens(extras, special_tokens=True)
    model.resize_token_embeddings(len(tok))
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write sid_to_item.pkl + sid_trie.pkl with a tiny 3-SID catalog."""
    sids = [
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (6, 7, 8, 9),
    ]
    trie = SIDTrie(sids, vocab_size=1024)
    sid_to_item = {sid: i + 1000 for i, sid in enumerate(sids)}
    sid_to_item_path = tmp_path / "sid_to_item.pkl"
    sid_trie_path = tmp_path / "sid_trie.pkl"
    with open(sid_to_item_path, "wb") as f:
        pickle.dump(sid_to_item, f)
    with open(sid_trie_path, "wb") as f:
        pickle.dump(trie, f)
    return sid_to_item_path, sid_trie_path, tmp_path


def _write_soft_prompt(path: Path) -> dict:
    """Write a soft_prompt.pt with deterministic weights and return the saved dict."""
    torch.manual_seed(11)
    weight = torch.randn(D_MODEL, QUERY_DIM)
    bias = torch.randn(D_MODEL)
    offsets = torch.randn(NUM_SOFT, D_MODEL)
    sd = {
        "query_projection.state_dict": OrderedDict([("weight", weight), ("bias", bias)]),
        "soft_prompt_offsets": offsets,
    }
    torch.save(sd, str(path))
    return sd


def test_load_retriever_with_soft_prompt(tmp_path):
    ckpt_dir = tmp_path / "hf_latest"
    ckpt_dir.mkdir()
    _make_tiny_t5_dir(ckpt_dir)
    sid_to_item_path, sid_trie_path, _ = _write_artifacts(tmp_path)
    saved = _write_soft_prompt(ckpt_dir / "soft_prompt.pt")

    retr = load_retriever(
        ckpt_dir=ckpt_dir,
        sid_to_item_path=sid_to_item_path,
        sid_trie_path=sid_trie_path,
        device="cpu",
    )

    # Soft-prompt fields populated.
    assert retr.query_projection is not None, "query_projection not loaded"
    assert retr.soft_prompt_offsets is not None, "soft_prompt_offsets not loaded"

    # Weights match what we wrote (allowing for device placement only).
    assert torch.allclose(
        retr.query_projection.weight.detach().cpu(),
        saved["query_projection.state_dict"]["weight"],
        atol=1e-6,
    )
    assert torch.allclose(
        retr.query_projection.bias.detach().cpu(),
        saved["query_projection.state_dict"]["bias"],
        atol=1e-6,
    )
    assert torch.allclose(
        retr.soft_prompt_offsets.detach().cpu(),
        saved["soft_prompt_offsets"],
        atol=1e-6,
    )

    # End-to-end search mode call works.
    q = torch.randn(QUERY_DIM)
    item_ids, sids = retr.retrieve_from_query_embedding(
        q, k=2, num_beams=4, constrained=True
    )
    assert len(item_ids) == 2 and len(sids) == 2


def test_load_retriever_without_soft_prompt_is_sequence_only(tmp_path):
    """No soft_prompt.pt -> retriever loads cleanly, search mode raises."""
    ckpt_dir = tmp_path / "hf_latest"
    ckpt_dir.mkdir()
    _make_tiny_t5_dir(ckpt_dir)
    sid_to_item_path, sid_trie_path, _ = _write_artifacts(tmp_path)
    # Note: deliberately NOT writing soft_prompt.pt.

    retr = load_retriever(
        ckpt_dir=ckpt_dir,
        sid_to_item_path=sid_to_item_path,
        sid_trie_path=sid_trie_path,
        device="cpu",
    )
    assert retr.query_projection is None
    assert retr.soft_prompt_offsets is None

    q = torch.randn(QUERY_DIM)
    with pytest.raises(RuntimeError):
        retr.retrieve_from_query_embedding(q, k=2, num_beams=4)


def test_explicit_soft_prompt_path_overrides_default(tmp_path):
    """If user passes `soft_prompt_path=` explicitly, that wins over the
    default `<ckpt_dir>/soft_prompt.pt` location."""
    ckpt_dir = tmp_path / "hf_latest"
    ckpt_dir.mkdir()
    _make_tiny_t5_dir(ckpt_dir)
    sid_to_item_path, sid_trie_path, _ = _write_artifacts(tmp_path)

    explicit_sp = tmp_path / "elsewhere" / "soft_prompt.pt"
    explicit_sp.parent.mkdir(parents=True)
    saved = _write_soft_prompt(explicit_sp)

    retr = load_retriever(
        ckpt_dir=ckpt_dir,
        sid_to_item_path=sid_to_item_path,
        sid_trie_path=sid_trie_path,
        soft_prompt_path=explicit_sp,
        device="cpu",
    )
    assert retr.query_projection is not None
    assert torch.allclose(
        retr.query_projection.weight.detach().cpu(),
        saved["query_projection.state_dict"]["weight"],
        atol=1e-6,
    )
