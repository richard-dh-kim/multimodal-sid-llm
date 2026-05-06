"""Search-mode eval (item CLIP embedding -> own SID) for a SID-LLM checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from sid_llm.eval.metrics import (
    hallucination_rate, ndcg_at_k, recall_at_k, silent_miss_rate,
)
from sid_llm.inference.beam_search import load_retriever


def _load_iid_to_sid(catalog_path: Path) -> dict[int, tuple[int, int, int, int]]:
    cat = pq.read_table(
        str(catalog_path),
        columns=["item_id", "sid_0", "sid_1", "sid_2", "sid_3"],
    )
    iids = cat.column("item_id").to_pylist()
    s0 = cat.column("sid_0").to_pylist()
    s1 = cat.column("sid_1").to_pylist()
    s2 = cat.column("sid_2").to_pylist()
    s3 = cat.column("sid_3").to_pylist()
    out: dict[int, tuple[int, int, int, int]] = {}
    for iid, a, b, c, d in zip(iids, s0, s1, s2, s3):
        if a is None:
            continue
        out[int(iid)] = (int(a), int(b), int(c), int(d))
    return out


def _load_embeddings(
    embeddings_path: Path,
    iid_to_sid: dict[int, tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[int]]:
    """(embeddings[N, D], item_ids[N]) for items present in both `embeddings_path` and `iid_to_sid`."""
    tbl = pq.read_table(str(embeddings_path), columns=["item_id", "embedding"])
    iids_full = tbl.column("item_id").to_pylist()
    embs_full = tbl.column("embedding").to_pylist()
    keep_iids: list[int] = []
    keep_embs: list[list[float]] = []
    for iid, emb in zip(iids_full, embs_full):
        iid_i = int(iid)
        if iid_i in iid_to_sid and emb is not None:
            keep_iids.append(iid_i)
            keep_embs.append(emb)
    if not keep_iids:
        raise RuntimeError(
            f"No embeddings in {embeddings_path} match items with SIDs in the catalog."
        )
    arr = np.asarray(keep_embs, dtype=np.float32)
    return arr, keep_iids


@click.command()
@click.option("--ckpt-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Retrieval-fine-tune output dir (contains soft_prompt.pt).")
@click.option("--embeddings-in", default="data/catalog/embeddings_b2.parquet",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Per-item CLIP embeddings parquet (item_id, embedding).")
@click.option("--catalog-in", default="data/catalog/catalog_with_sid.parquet",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sid-to-item", default="data/catalog/sid_to_item.pkl",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sid-trie", default="data/catalog/sid_trie.pkl",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--soft-prompt-path", default=None,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Override the default <ckpt-dir>/soft_prompt.pt location.")
@click.option("--out", default="eval/results/sid_llm_search.json",
              type=click.Path(dir_okay=False, path_type=Path))
@click.option("--ks", default="10,50,100", type=str)
@click.option("--max-queries", default=2_000, type=int,
              help="Beam search is slow; default cap mirrors the text eval CLI.")
@click.option("--num-beams", default=10, type=int)
@click.option("--constrained/--unconstrained", default=True)
@click.option("--baseline-name", default="SID_LLM_search", type=str)
@click.option("--seed", default=42, type=int)
@click.option("--device", default=None, type=str)
def main(
    ckpt_dir, embeddings_in, catalog_in, sid_to_item, sid_trie, soft_prompt_path,
    out, ks, max_queries, num_beams, constrained, baseline_name, seed, device,
):
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading retriever from {ckpt_dir} ...")
    retr = load_retriever(
        ckpt_dir, sid_to_item, sid_trie,
        soft_prompt_path=soft_prompt_path, device=device,
    )
    print(f"  device={retr.device}")
    if retr.query_projection is None:
        raise click.ClickException(
            f"{ckpt_dir} has no soft_prompt.pt; this CLI requires a search-mode "
            f"checkpoint (try --soft-prompt-path explicitly, or run the text-mode "
            f"eval `sid_llm_eval` instead)."
        )

    print(f"Loading catalog SIDs from {catalog_in} ...")
    iid_to_sid = _load_iid_to_sid(catalog_in)
    print(f"  {len(iid_to_sid):,} items have SIDs")

    print(f"Loading embeddings from {embeddings_in} ...")
    embs, iids = _load_embeddings(embeddings_in, iid_to_sid)
    print(f"  {len(iids):,} items have both an embedding and a SID")

    rng = np.random.default_rng(seed)
    n_avail = len(iids)
    n_q = min(max_queries, n_avail)
    sample_idx = rng.choice(n_avail, size=n_q, replace=False)
    queries_emb = embs[sample_idx]                                 # [n_q, 512]
    target_iids = [iids[i] for i in sample_idx]
    print(f"  sampled {n_q:,} query items")

    k_list = [int(x) for x in ks.split(",")]
    max_k = max(k_list)
    if num_beams < max_k:
        print(
            f"WARNING: num_beams ({num_beams}) < max(ks) ({max_k}); "
            f"auto-bumping num_beams to {max_k}."
        )
        num_beams = max_k

    print(
        f"Running search-mode beam search over {n_q:,} queries "
        f"(num_beams={num_beams}, constrained={constrained}) ..."
    )
    preds: list[list[int]] = []
    top1_sids: list[tuple[int, ...]] = []
    for q_emb in tqdm(queries_emb, desc="retrieve"):
        q = torch.from_numpy(q_emb).float()
        item_ids, sids = retr.retrieve_from_query_embedding(
            q, k=max_k, num_beams=num_beams, constrained=constrained
        )
        # Push silent-miss (-1) to the back so it doesn't displace hits in recall@k.
        preds.append(
            [iid for iid in item_ids if iid >= 0]
            + [iid for iid in item_ids if iid < 0]
        )
        if sids:
            top1_sids.append(sids[0])

    valid_sids = set(retr.sid_to_item.keys())
    halluc = hallucination_rate(top1_sids, valid_sids)
    silent = silent_miss_rate(top1_sids, retr.sid_to_item)

    results = {f"recall@{k}": recall_at_k(preds, target_iids, k) for k in k_list}
    results.update({f"ndcg@{k}": ndcg_at_k(preds, target_iids, k) for k in k_list})
    results["hallucination_rate"] = halluc
    results["silent_miss_rate"] = silent

    print(f"\n{baseline_name} results (search mode, item->own-SID task):")
    for kk, vv in results.items():
        print(f"  {kk}: {vv:.4f}")

    out.write_text(json.dumps({
        "baseline": baseline_name,
        "task": "item-clip-embedding-to-own-sid",
        "n_queries": n_q,
        "num_beams": num_beams,
        "constrained": constrained,
        "metrics": results,
    }, indent=2))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
