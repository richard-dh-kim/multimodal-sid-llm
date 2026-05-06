"""CPU smoke: RetrievalLightning search-mode forward + backward reaches the soft-prompt params."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast


@pytest.fixture(scope="module")
def tiny_init_dir(tmp_path_factory) -> Path:
    """Tiny T5 + tokenizer on disk (with <search> token added) for RetrievalLightning to load."""
    out = tmp_path_factory.mktemp("tiny_init")
    cfg = T5Config(
        vocab_size=32128,
        d_model=64,
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
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return out


def test_search_mode_smoke_grad_reaches_soft_prompt(tiny_init_dir):
    from sid_llm.training.train_retrieval import RetrievalLightning

    module = RetrievalLightning(
        init_dir=tiny_init_dir,
        soft_prompt_path=None,
        lr=1e-4,
        weight_decay=0.0,
        gradient_checkpointing=False,
        use_anchored=False,
    )
    module.train()
    module.to("cpu")
    module.float()

    assert module._search_token_id != module.tokenizer.unk_token_id, \
        "<search> token missing from tokenizer; smoke fixture is broken."

    bs = 3
    sid_eos_id = module.tokenizer.convert_tokens_to_ids("<sid_eos>")
    sid_0_id = module.tokenizer.convert_tokens_to_ids("<sid_0>")
    labels = torch.full((bs, 5), fill_value=sid_0_id, dtype=torch.long)
    labels[:, -1] = sid_eos_id

    batch = {
        "mode": "search",
        "query_embeddings": torch.randn(bs, 512),
        "labels": labels,
    }

    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()

    qp_grad = module.query_projection.weight.grad
    off_grad = module.soft_prompt_offsets.grad
    assert qp_grad is not None, "query_projection.weight.grad is None — soft prompt detached!"
    assert off_grad is not None, "soft_prompt_offsets.grad is None — soft prompt detached!"
    assert torch.isfinite(qp_grad).all()
    assert torch.isfinite(off_grad).all()
    assert qp_grad.abs().sum().item() > 0, "query_projection.weight grad is exactly zero"
    assert off_grad.abs().sum().item() > 0, "soft_prompt_offsets grad is exactly zero"


def test_sequence_mode_smoke(tiny_init_dir):
    """Sequence-mode batches go through the input_ids path; soft-prompt params receive no grad."""
    from sid_llm.training.train_retrieval import RetrievalLightning

    module = RetrievalLightning(
        init_dir=tiny_init_dir,
        soft_prompt_path=None,
        lr=1e-4,
        weight_decay=0.0,
        gradient_checkpointing=False,
        use_anchored=False,
    )
    module.train()
    module.to("cpu")
    module.float()

    seq_id = module.tokenizer.convert_tokens_to_ids("<seq>")
    sid_eos_id = module.tokenizer.convert_tokens_to_ids("<sid_eos>")
    sid_0_id = module.tokenizer.convert_tokens_to_ids("<sid_0>")

    bs = 2
    input_ids = torch.tensor([[seq_id, sid_0_id, sid_0_id]] * bs, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[sid_0_id, sid_0_id, sid_eos_id]] * bs, dtype=torch.long)

    batch = {
        "mode": "sequence",
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
    loss.backward()
    # Soft-prompt params are not on the input_ids path, so no grad.
    assert module.query_projection.weight.grad is None or \
        module.query_projection.weight.grad.abs().sum().item() == 0


def test_load_soft_prompt_from_real_init():
    init_path = Path("checkpoints/sid_llm/init/soft_prompt.pt")
    if not init_path.exists():
        pytest.skip(f"{init_path} not present; load test skipped.")
    from sid_llm.training.train_retrieval import RetrievalLightning

    real_init = Path("checkpoints/sid_llm/init/hf_model")
    if not real_init.exists():
        pytest.skip(f"{real_init} not present; load test skipped.")

    module = RetrievalLightning(
        init_dir=real_init,
        soft_prompt_path=init_path,
        lr=1e-4,
        weight_decay=0.0,
        gradient_checkpointing=False,
        use_anchored=False,
    )
    sd = torch.load(str(init_path), map_location="cpu", weights_only=False)
    expected_w = sd["query_projection.state_dict"]["weight"]
    expected_offsets = sd["soft_prompt_offsets"]
    assert torch.allclose(
        module.query_projection.weight.detach().cpu(), expected_w, atol=1e-6
    ), "query_projection.weight not loaded from soft_prompt.pt"
    assert torch.allclose(
        module.soft_prompt_offsets.detach().cpu(), expected_offsets, atol=1e-6
    ), "soft_prompt_offsets not loaded from soft_prompt.pt"
