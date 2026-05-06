"""TIGER baseline (Rajput et al. 2023, Sec. 4.2): small T5 from scratch over the SID vocab.

Reuses the M3.5 tokenizer (vocab parity with SID-LLM keeps the eval harness
drop-in). Trained on behavior rows only.
"""
from __future__ import annotations

from pathlib import Path

from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast


TIGER_NUM_LAYERS = 6
TIGER_NUM_DECODER_LAYERS = 6
TIGER_D_MODEL = 128
TIGER_D_FF = 1024
TIGER_NUM_HEADS = 4
TIGER_D_KV = TIGER_D_MODEL // TIGER_NUM_HEADS
TIGER_DROPOUT = 0.1


def build_tiger_baseline(
    tokenizer_path: Path,
) -> tuple[T5TokenizerFast, T5ForConditionalGeneration]:
    """Build a small T5 (TIGER recipe) with random weights and the M3.5 tokenizer."""
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
        # T5-base structural defaults; only dimensions shrink.
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
