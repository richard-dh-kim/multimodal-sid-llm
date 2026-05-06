"""Unit + smoke tests for the TIGER replication baseline (B3).

Three tests:
  1. `test_build_tiger_baseline_dim` — model has the published TIGER dims and
     a vocab matching the M3.5 expanded tokenizer; total parameter count is
     small (under 30M, dominated by the 33k-vocab embedding table).
  2. `test_train_tiger_dataset_filters_to_behavior_rows` — the dataset class
     drops metadata rows.
  3. `test_train_tiger_smoke_step` — one CPU train step on synthetic data:
     loss is finite, encoder gradient is non-zero.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from sid_llm.models.tiger_baseline import (
    TIGER_D_FF,
    TIGER_D_KV,
    TIGER_D_MODEL,
    TIGER_NUM_DECODER_LAYERS,
    TIGER_NUM_HEADS,
    TIGER_NUM_LAYERS,
    build_tiger_baseline,
)
from sid_llm.training.train_tiger import T5Collator, TigerBehaviorDataset, TigerLightning


_INIT_DIR = Path("checkpoints/sid_llm/init/hf_model")


@pytest.mark.skipif(not _INIT_DIR.exists(), reason="M3.5 init dir not present")
def test_build_tiger_baseline_dim():
    tokenizer, model = build_tiger_baseline(_INIT_DIR)
    cfg = model.config
    assert cfg.d_model == TIGER_D_MODEL == 128
    assert cfg.d_ff == TIGER_D_FF == 1024
    assert cfg.d_kv == TIGER_D_KV == 32
    assert cfg.num_layers == TIGER_NUM_LAYERS == 6
    assert cfg.num_decoder_layers == TIGER_NUM_DECODER_LAYERS == 6
    assert cfg.num_heads == TIGER_NUM_HEADS == 4
    # Vocab parity with the shared M3.5 tokenizer is what makes the eval
    # harness drop in across baselines.
    assert cfg.vocab_size == len(tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 30_000_000, f"TIGER baseline grew to {n_params:,} params"


def test_train_tiger_dataset_filters_to_behavior_rows(tmp_path: Path):
    rows = [
        {"seq_type": "metadata", "input_text": "<seq> title: x",
         "target_text": "<sid_1><sid_2><sid_eos>"},
        {"seq_type": "behavior", "input_text": "<seq> <sid_3>",
         "target_text": "<sid_4><sid_eos>"},
        {"seq_type": "metadata", "input_text": "<seq> brand: y",
         "target_text": "<sid_5><sid_eos>"},
        {"seq_type": "behavior", "input_text": "<seq> <sid_6><sid_7>",
         "target_text": "<sid_8><sid_eos>"},
    ]
    p = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(p))
    ds = TigerBehaviorDataset(p)
    assert len(ds) == 2, "metadata rows should be filtered out"
    inputs = {ds[i]["input_text"] for i in range(len(ds))}
    assert inputs == {"<seq> <sid_3>", "<seq> <sid_6><sid_7>"}


@pytest.mark.skipif(not _INIT_DIR.exists(), reason="M3.5 init dir not present")
def test_train_tiger_smoke_step(tmp_path: Path):
    """One forward+backward on CPU. Asserts loss is finite and grads reach the
    encoder body (i.e. autograd is wired through the TIGER model)."""
    # Tiny synthetic behavior corpus.
    rows = [
        {"seq_type": "behavior",
         "input_text": "<seq> <sid_a_1><sid_b_2><sid_c_3><sid_d_4>",
         "target_text": "<sid_a_5><sid_b_6><sid_c_7><sid_d_8><sid_eos>"}
        for _ in range(8)
    ]
    p = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(p))
    ds = TigerBehaviorDataset(p)
    assert len(ds) == 8

    pl_module = TigerLightning(
        tokenizer_path=_INIT_DIR,
        lr=1e-3,
        weight_decay=0.01,
        warmup_frac=0.3,
        total_steps=4,
        gradient_checkpointing=False,
    )
    pl_module.train()
    collate = T5Collator(pl_module.tokenizer, max_input_len=64, max_target_len=16)
    batch = collate([ds[i] for i in range(4)])
    out = pl_module.model(**batch)
    assert torch.isfinite(out.loss), f"loss not finite: {out.loss.item()}"
    out.loss.backward()
    # The first encoder block's input layer norm is a small-but-nonzero leaf
    # near the input — its grad being nonzero proves the chain reached the
    # body of the TIGER stack (not just the embedding table).
    enc_block_param = next(
        p for n, p in pl_module.model.named_parameters()
        if n.startswith("encoder.block.0.layer.0") and p.requires_grad
    )
    assert enc_block_param.grad is not None
    assert enc_block_param.grad.abs().sum().item() > 0
