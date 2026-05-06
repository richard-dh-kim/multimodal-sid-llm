"""CPU-only smoke test for RetrievalLightning.search-mode forward + backward.

Guards against the soft-prompt being detached from the autograd graph: after
one training_step on a search-mode batch we expect a finite loss AND
non-None / non-zero gradients on `query_projection.weight` and
`soft_prompt_offsets`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast


@pytest.fixture(scope="module")
def tiny_init_dir(tmp_path_factory) -> Path:
    """Build a minimal T5 + tokenizer on disk so RetrievalLightning can load it.

    Avoids hitting the network: we instantiate from a small T5Config with
    `num_layers=1, d_model=64, d_ff=128, num_heads=2`. We also expand the
    tokenizer with the same special tokens M3.5 adds (1024 sids + control +
    mode markers) so `<search>` resolves to a real token id.
    """
    out = tmp_path_factory.mktemp("tiny_init")
    cfg = T5Config(
        vocab_size=32128,                # default t5-small vocab; we'll resize
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
        soft_prompt_path=None,            # use random init
        lr=1e-4,
        weight_decay=0.0,
        gradient_checkpointing=False,
        use_anchored=False,
    )
    module.train()
    # Force CPU + float32 to keep this a pure-CPU test.
    module.to("cpu")
    module.float()

    # Confirm <search> resolved to a real token id, not unk.
    assert module._search_token_id != module.tokenizer.unk_token_id, \
        "<search> token missing from tokenizer; smoke fixture is broken."

    bs = 3
    sid_eos_id = module.tokenizer.convert_tokens_to_ids("<sid_eos>")
    sid_0_id = module.tokenizer.convert_tokens_to_ids("<sid_0>")
    # Build a tiny labels tensor of shape [bs, 5]: 4 SID tokens + eos.
    labels = torch.full((bs, 5), fill_value=sid_0_id, dtype=torch.long)
    labels[:, -1] = sid_eos_id

    batch = {
        "mode": "search",
        "query_embeddings": torch.randn(bs, 512),
        "labels": labels,
    }

    # Zero grads, run one training_step, backprop, check grads.
    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()

    # Soft-prompt params must receive grad.
    qp_grad = module.query_projection.weight.grad
    off_grad = module.soft_prompt_offsets.grad
    assert qp_grad is not None, "query_projection.weight.grad is None — soft prompt detached!"
    assert off_grad is not None, "soft_prompt_offsets.grad is None — soft prompt detached!"
    assert torch.isfinite(qp_grad).all()
    assert torch.isfinite(off_grad).all()
    assert qp_grad.abs().sum().item() > 0, "query_projection.weight grad is exactly zero"
    assert off_grad.abs().sum().item() > 0, "soft_prompt_offsets grad is exactly zero"


def test_sequence_mode_smoke(tiny_init_dir):
    """Sanity: sequence-mode batches still go through the input_ids path."""
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
    # In sequence mode, soft-prompt params should NOT receive grad — they
    # weren't on the path. (T5 forward through input_ids only.)
    assert module.query_projection.weight.grad is None or \
        module.query_projection.weight.grad.abs().sum().item() == 0


def test_load_soft_prompt_from_real_init():
    """If the M3.5 init file exists on disk, loading it must succeed and
    populate query_projection + soft_prompt_offsets."""
    init_path = Path("checkpoints/sid_llm/init/soft_prompt.pt")
    if not init_path.exists():
        pytest.skip(f"{init_path} not present; load test skipped.")
    # Also need a tiny T5 init dir; build one inline.
    from sid_llm.training.train_retrieval import RetrievalLightning

    # We need an init_dir with a real T5 + tokenizer that has <search>. Use
    # the M3.5 init checkpoint if it's around; otherwise skip.
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
    # Re-load and compare to the on-disk state to confirm the load took effect.
    sd = torch.load(str(init_path), map_location="cpu", weights_only=False)
    expected_w = sd["query_projection.state_dict"]["weight"]
    expected_offsets = sd["soft_prompt_offsets"]
    assert torch.allclose(
        module.query_projection.weight.detach().cpu(), expected_w, atol=1e-6
    ), "query_projection.weight not loaded from soft_prompt.pt"
    assert torch.allclose(
        module.soft_prompt_offsets.detach().cpu(), expected_offsets, atol=1e-6
    ), "soft_prompt_offsets not loaded from soft_prompt.pt"
