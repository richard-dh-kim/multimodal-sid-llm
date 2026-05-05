"""Baselines: B0 random, B1 plain-CLIP MIPS, B2 VL-CLIP MIPS, B3 TIGER (later), B4 HSTU (later)."""
from __future__ import annotations

import json
from itertools import groupby
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq

from sid_llm.eval.metrics import ndcg_at_k, recall_at_k


def mips_topk(
    queries: np.ndarray, catalog_emb: np.ndarray,
    catalog_item_ids: list[int], k: int,
) -> list[list[int]]:
    """Brute-force MIPS. queries [N,D] x catalog_emb.T [D,M] -> [N,M]; take top-K per row."""
    scores = queries @ catalog_emb.T  # [N, M]
    topk_idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    rows = np.arange(len(queries))[:, None]
    topk_sorted = topk_idx[rows, np.argsort(-scores[rows, topk_idx])]
    return [[catalog_item_ids[i] for i in row] for row in topk_sorted]


def _build_user_history_queries(
    interactions: list[dict],
    catalog_emb_lookup: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Leave-last-out per user. Input = mean of user's prior item embeddings;
    target = last item. Skips users with <2 catalog-resolved events.

    Simple v0 query construction; replaced in M3 with the real SID-LLM input.
    """
    interactions.sort(key=lambda r: (r["user_id"], r["timestamp_ms"]))

    queries: list[np.ndarray] = []
    targets: list[int] = []
    for _uid, group in groupby(interactions, key=lambda r: r["user_id"]):
        events = list(group)
        if len(events) < 2:
            continue
        history_embs: list[np.ndarray] = []
        for ev in events[:-1]:
            iid = ev.get("_resolved_item_id")
            if iid is not None and iid in catalog_emb_lookup:
                history_embs.append(catalog_emb_lookup[iid])
        last_iid = events[-1].get("_resolved_item_id")
        if not history_embs or last_iid is None or last_iid not in catalog_emb_lookup:
            continue
        q = np.mean(np.stack(history_embs), axis=0)
        q = q / (np.linalg.norm(q) + 1e-9)
        queries.append(q.astype(np.float32))
        targets.append(int(last_iid))

    if not queries:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack(queries), targets


@click.command()
@click.option(
    "--catalog-in", default="data/catalog/catalog.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--embeddings-in", default="data/catalog/embeddings_b1.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--interactions-in", default="data/catalog/interactions.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out", default="eval/results/baseline_b1.json",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--ks", default="10,50,100", type=str)
@click.option(
    "--baseline-name", default="B1_plain_clip_mips", type=str,
    help="Tag written into the results JSON (used for B2/B3/B4 reuse).",
)
def main(
    catalog_in: Path, embeddings_in: Path, interactions_in: Path,
    out: Path, ks: str, baseline_name: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading catalog and embeddings ...")
    catalog = pq.read_table(str(catalog_in))
    embs = pq.read_table(str(embeddings_in))
    item_ids = embs.column("item_id").to_pylist()
    emb_arr = np.array(embs.column("embedding").to_pylist(), dtype=np.float32)
    print(f"  catalog={catalog.num_rows:,} items, embeddings={len(item_ids):,}")

    asin_to_iid = dict(zip(
        catalog.column("parent_asin").to_pylist(),
        catalog.column("item_id").to_pylist(),
    ))

    print("Loading interactions ...")
    inter = pq.read_table(str(interactions_in)).to_pylist()
    for r in inter:
        r["_resolved_item_id"] = asin_to_iid.get(r["parent_asin"])

    catalog_emb_lookup = {
        int(iid): emb_arr[i] for i, iid in enumerate(item_ids)
    }
    queries, targets = _build_user_history_queries(inter, catalog_emb_lookup)
    print(f"  built {len(queries):,} test queries (leave-last-out)")

    if len(queries) == 0:
        print("WARNING: no queries built; nothing to evaluate.")
        return

    k_list = [int(x) for x in ks.split(",")]
    max_k = max(k_list)
    print(f"Running MIPS top-{max_k} over {len(item_ids):,} catalog items ...")
    preds = mips_topk(queries, emb_arr, item_ids, k=max_k)

    results: dict[str, float] = {}
    for k in k_list:
        results[f"recall@{k}"] = recall_at_k(preds, targets, k)
        results[f"ndcg@{k}"] = ndcg_at_k(preds, targets, k)

    print(f"\n{baseline_name} results:")
    for kk, vv in results.items():
        print(f"  {kk}: {vv:.4f}")

    out.write_text(json.dumps(
        {"baseline": baseline_name, "n_queries": len(queries), "metrics": results},
        indent=2,
    ))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
