"""Diagnostic: trace one CPT training example end-to-end through the loaded checkpoint."""
from __future__ import annotations

import pickle
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast


CKPT = Path("checkpoints/sid_llm/cpt/hf_best")
CORPUS = Path("data/catalog/cpt_corpus.parquet")


def main():
    print(f"Loading model+tokenizer from {CKPT}")
    tok = T5TokenizerFast.from_pretrained(str(CKPT))
    model = T5ForConditionalGeneration.from_pretrained(str(CKPT))
    model.eval().to("cuda")

    print("Tokenizer vocab size:", len(tok))
    print("Model vocab size:    ", model.config.vocab_size)
    print("model.shared.weight.shape:", tuple(model.shared.weight.shape))
    print("decoder_start_token_id:", model.config.decoder_start_token_id)
    print("pad_token_id:          ", model.config.pad_token_id)
    print("eos_token_id:          ", model.config.eos_token_id)

    # SID token ID resolution
    sid_token_ids = tok.convert_tokens_to_ids([f"<sid_{i}>" for i in range(1024)])
    sid_eos_id = tok.convert_tokens_to_ids("<sid_eos>")
    print(f"<sid_0>   -> {sid_token_ids[0]}")
    print(f"<sid_1>   -> {sid_token_ids[1]}")
    print(f"<sid_1023>-> {sid_token_ids[-1]}")
    print(f"<sid_eos> -> {sid_eos_id}")
    none_ids = [i for i, x in enumerate(sid_token_ids) if x in (None, tok.unk_token_id)]
    print(f"  unresolved sid token IDs: {len(none_ids)} (first few={none_ids[:5]})")

    print("\nLoading corpus")
    table = pq.read_table(str(CORPUS))
    # Pick a metadata row
    md_idx = None
    for i in range(table.num_rows):
        st = table.column("seq_type")[i].as_py()
        if st == "metadata":
            md_idx = i
            break
    row = table.slice(md_idx, 1).to_pylist()[0]
    print(f"\n--- sample row [{md_idx}] ---")
    print("input_text :", row["input_text"][:200])
    print("target_text:", row["target_text"])

    # Tokenize input + target
    inp = tok(row["input_text"], return_tensors="pt", truncation=True, max_length=512).to("cuda")
    tgt = tok(row["target_text"], return_tensors="pt", padding=False, truncation=True, max_length=16).to("cuda")
    labels = tgt["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100

    print(f"\ninput_ids[0] (first 30): {inp['input_ids'][0][:30].tolist()}")
    print(f"target_ids[0]         : {tgt['input_ids'][0].tolist()}")
    target_decoded = tok.convert_ids_to_tokens(tgt["input_ids"][0].tolist())
    print(f"target as tokens      : {target_decoded}")

    with torch.no_grad():
        out = model(**inp, labels=labels)
    print(f"\nForward-pass loss on this training example: {out.loss.item():.4f}")
    print("(if CPT worked, this should be small; random would be ~10+)")

    print("\n--- Now: unconstrained greedy generate ---")
    with torch.no_grad():
        gen = model.generate(
            **inp, max_new_tokens=8, do_sample=False, num_beams=1,
            return_dict_in_generate=True,
        )
    print("generated ids:", gen.sequences[0].tolist())
    print("decoded tokens:", tok.convert_ids_to_tokens(gen.sequences[0].tolist()))
    print("decoded text :", tok.decode(gen.sequences[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
