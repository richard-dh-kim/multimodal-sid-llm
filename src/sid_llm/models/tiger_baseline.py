"""TIGER replication baseline (B3): vanilla TIGER recipe over our SID vocabulary.

This baseline strips every contribution that is specific to the SID-LLM project
and trains a small T5 from scratch on the same Semantic-ID corpus. It exists
solely so the README ablation table can quote "our additions add X recall@10
over a TIGER-style baseline."

What is stripped vs. SID-LLM (M3.6+M3.7):
  - No CPT init: model is randomly initialized, no `from_pretrained`.
  - No multimodal front-end: text-only, no image/CLIP tower, no soft prompts.
  - No AdamWAnchored: plain `torch.optim.AdamW`.
  - No Trie / constrained decoding at training time.
  - No KV-cache ablation hooks (that ablation is inference-time only).

What is shared with SID-LLM:
  - The expanded T5 tokenizer at `checkpoints/sid_llm/init/hf_model` (33,128 tokens
    including all SIDs + `<seq>` + `<sid_eos>`). This keeps `sid_llm_eval.py`
    drop-in compatible — vocab IDs match across baselines.
  - The same `cpt_corpus.parquet`, behavior rows only (TIGER does not train on
    metadata-style sequences).

Architecture follows the published TIGER recipe (Rajput et al., 2023, Sec. 4.2):
  num_layers=6 encoder, num_decoder_layers=6, d_model=128, d_ff=1024, num_heads=4.
  d_kv = d_model / num_heads = 32. ~4-6M trainable params depending on vocab
  (most of the parameter mass is the input/output embedding table over the
  33k-token vocab; tie_word_embeddings=True keeps that to one copy).
"""
from __future__ import annotations

from pathlib import Path

from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast


# Public TIGER recipe — keep these as module-level constants so the test file
# can import them and assert the trained baseline matches.
TIGER_NUM_LAYERS = 6
TIGER_NUM_DECODER_LAYERS = 6
TIGER_D_MODEL = 128
TIGER_D_FF = 1024
TIGER_NUM_HEADS = 4
TIGER_D_KV = TIGER_D_MODEL // TIGER_NUM_HEADS  # 32
TIGER_DROPOUT = 0.1


def build_tiger_baseline(
    tokenizer_path: Path,
) -> tuple[T5TokenizerFast, T5ForConditionalGeneration]:
    """Construct a small T5 matching TIGER's published recipe, randomly initialized.

    Args:
        tokenizer_path: directory containing the M3.5 expanded tokenizer (the
            `hf_model` dir under `checkpoints/sid_llm/init/`). We only borrow
            the tokenizer; the model itself is built from scratch.

    Returns:
        (tokenizer, model) where the model has random weights and a vocab
        matching `len(tokenizer)`.
    """
    tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_path))
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=TIGER_D_MODEL,
        d_ff=TIGER_D_FF,
        d_kv=TIGER_D_KV,
        num_layers=TIGER_NUM_LAYERS,
        num_decoder_layers=TIGER_NUM_DECODER_LAYERS,
        num_heads=TIGER_NUM_HEADS,
        dropout_rate=TIGER_DROPOUT,
        # Match T5-base structural defaults so the HF save/load + `sid_llm_eval`
        # generation path behaves identically — only the dimensions shrink.
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1,
        decoder_start_token_id=0,
        use_cache=True,
    )
    model = T5ForConditionalGeneration(config)
    return tokenizer, model
