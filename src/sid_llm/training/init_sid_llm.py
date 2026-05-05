"""CLI: build the initialized SID-LLM checkpoint and save to disk.

Usage:
  python -m sid_llm.training.init_sid_llm --model-id t5-base --out checkpoints/sid_llm/init
"""
from __future__ import annotations

from pathlib import Path

import click
import torch

from sid_llm.models.sid_llm import SIDLLMLightning, SID_TOKEN_NAMES


@click.command()
@click.option("--model-id", default="t5-base", type=str)
@click.option(
    "--out", default="checkpoints/sid_llm/init",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--query-embed-dim", default=512, type=int)
@click.option("--num-soft-prompt-tokens", default=4, type=int)
@click.option("--seed", default=42, type=int)
def main(model_id: str, out: Path, query_embed_dim: int, num_soft_prompt_tokens: int, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    print(f"Building SID-LLM from {model_id} ...")
    m = SIDLLMLightning(
        model_id=model_id,
        query_embed_dim=query_embed_dim,
        num_soft_prompt_tokens=num_soft_prompt_tokens,
    )

    new_vocab = m.model.config.vocab_size
    n_params = sum(p.numel() for p in m.model.parameters())
    print(f"  expanded vocab: {new_vocab:,}  ({len(SID_TOKEN_NAMES):,} new tokens)")
    print(f"  T5 model params: {n_params:,}")
    print(f"  query_projection: {m.query_projection.in_features} -> {m.query_projection.out_features}")
    print(f"  soft_prompt_offsets: {tuple(m.soft_prompt_offsets.shape)}")

    # Save HF-format model + tokenizer.
    hf_dir = out / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    m.model.save_pretrained(str(hf_dir))
    m.tokenizer.save_pretrained(str(hf_dir))
    print(f"\nSaved HF model + tokenizer -> {hf_dir}")

    # Save the soft-prompt machinery (NOT part of HF model state).
    extras = {
        "query_projection.state_dict": m.query_projection.state_dict(),
        "soft_prompt_offsets": m.soft_prompt_offsets.detach().cpu(),
        "sid_token_ids": m.sid_token_ids,
        "sid_pad_id": m.sid_pad_id,
        "sid_eos_id": m.sid_eos_id,
        "seq_id": m.seq_id,
        "search_id": m.search_id,
    }
    torch.save(extras, str(out / "soft_prompt.pt"))
    print(f"Saved soft-prompt extras    -> {out / 'soft_prompt.pt'}")

    # Sanity output: id ranges
    print(f"\nSID token id range: [{m.sid_token_ids[0]:,}, {m.sid_token_ids[-1]:,}]")
    print(f"sid_pad_id={m.sid_pad_id}  sid_eos_id={m.sid_eos_id}")
    print(f"seq_id={m.seq_id}  search_id={m.search_id}")


if __name__ == "__main__":
    main()
