"""CPT-aligned sanity eval: samples from cpt_corpus.parquet and reports per-seq_type recall@1/k."""
from __future__ import annotations

import random
from pathlib import Path

import click
import pyarrow.parquet as pq
import torch
from transformers import LogitsProcessorList

from sid_llm.inference.beam_search import load_retriever
from sid_llm.inference.logits_processor import TrieConstrainedSIDProcessor


def _target_sid_tuple(target_text: str, token_id_to_codebook_index: dict[int, int],
                      tokenizer) -> tuple[int, int, int, int] | None:
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    cb: list[int] = []
    for tid in ids:
        if tid in token_id_to_codebook_index:
            cb.append(token_id_to_codebook_index[tid])
            if len(cb) == 4:
                return tuple(cb)
    return None


@click.command()
@click.option("--ckpt-dir", default="checkpoints/sid_llm/cpt/hf_best",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--corpus-in", default="data/catalog/cpt_corpus.parquet",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sid-to-item", default="data/catalog/sid_to_item.pkl",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sid-trie", default="data/catalog/sid_trie.pkl",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--n", default=50, type=int, help="Samples per seq_type.")
@click.option("--num-beams", default=8, type=int)
@click.option("--seed", default=42, type=int)
def main(ckpt_dir, corpus_in, sid_to_item, sid_trie, n, num_beams, seed):
    print(f"Loading retriever from {ckpt_dir} ...")
    retr = load_retriever(ckpt_dir, sid_to_item, sid_trie)
    print(f"  device={retr.device}")

    print(f"Loading corpus {corpus_in} ...")
    table = pq.read_table(str(corpus_in))
    print(f"  {table.num_rows:,} rows")

    rng = random.Random(seed)
    seq_types = sorted(set(table.column("seq_type").to_pylist()))
    print(f"  seq_types: {seq_types}")

    summary: dict[str, dict[str, float]] = {}
    for st in seq_types:
        mask = [t == st for t in table.column("seq_type").to_pylist()]
        idxs = [i for i, m in enumerate(mask) if m]
        if not idxs:
            continue
        sampled = rng.sample(idxs, min(n, len(idxs)))
        rows = [table.slice(i, 1).to_pylist()[0] for i in sampled]

        print(f"\n--- seq_type='{st}' ({len(rows)} samples) ---")

        top1_hits = 0
        topk_hits = 0
        valid_targets = 0
        for r in rows:
            tgt_tup = _target_sid_tuple(r["target_text"], retr.token_id_to_codebook_index, retr.tokenizer)
            if tgt_tup is None:
                continue
            valid_targets += 1

            enc = retr.tokenizer(r["input_text"], return_tensors="pt",
                                 truncation=True, max_length=512).to(retr.device)
            logits_processors = LogitsProcessorList([
                TrieConstrainedSIDProcessor(retr.trie, retr.sid_token_ids, decoder_start_offset=1),
            ])
            with torch.no_grad():
                out = retr.model.generate(
                    **enc, max_new_tokens=5, num_beams=num_beams,
                    num_return_sequences=num_beams,
                    do_sample=False, logits_processor=logits_processors,
                    return_dict_in_generate=True, use_cache=True,
                )
            preds = retr._decode_sid_tuples_from_sequences(out.sequences)
            if preds and preds[0] == tgt_tup:
                top1_hits += 1
            if tgt_tup in preds[:num_beams]:
                topk_hits += 1

        recall1 = top1_hits / max(1, valid_targets)
        recallk = topk_hits / max(1, valid_targets)
        print(f"  valid_targets={valid_targets}/{len(rows)}")
        print(f"  recall@1     = {recall1:.4f}")
        print(f"  recall@{num_beams:<3}  = {recallk:.4f}")
        summary[st] = {"recall@1": recall1, f"recall@{num_beams}": recallk,
                       "n": valid_targets}

    print("\n=== summary ===")
    for st, m in summary.items():
        print(f"  {st:10s}  recall@1={m['recall@1']:.4f}  recall@{num_beams}={m[f'recall@{num_beams}']:.4f}  (n={m['n']})")


if __name__ == "__main__":
    main()
