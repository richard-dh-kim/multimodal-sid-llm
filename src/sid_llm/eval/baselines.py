"""Baselines: B0 random, B1 plain-CLIP MIPS, B2 VL-CLIP MIPS, B3 TIGER (later), B4 HSTU (later)."""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sid_llm.eval.metrics import ndcg_at_k, recall_at_k


def mips_topk(
    queries: np.ndarray, catalog_emb: np.ndarray,
    catalog_item_ids: list[int], k: int,
    chunk_size: int = 1024,
) -> list[list[int]]:
    """Brute-force MIPS. Chunked over queries to bound peak memory.
    For each chunk: scores [chunk, M] then top-K per row; concatenate chunks.

    queries: [N, D]; catalog_emb: [M, D]; returns list of K item_ids per query.
    """
    n = len(queries)
    if n == 0:
        return []
    cat_t = catalog_emb.T  # [D, M], not copied per chunk
    m = catalog_emb.shape[0]
    k_eff = min(k, m)
    out: list[list[int]] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        scores = queries[start:end] @ cat_t  # [chunk, M]
        # argpartition for unordered top-K, then sort the K indices by score
        part_kth = min(k_eff, scores.shape[1] - 1)
        topk_idx = np.argpartition(-scores, kth=part_kth, axis=1)[:, :k_eff]
        rows = np.arange(end - start)[:, None]
        topk_sorted = topk_idx[rows, np.argsort(-scores[rows, topk_idx])]
        for row in topk_sorted:
            out.append([catalog_item_ids[i] for i in row])
    return out


def _build_user_history_queries(
    interactions_path: Path,
    asin_to_iid: dict[str, int],
    catalog_emb_arr: np.ndarray,
    iid_to_emb_idx: dict[int, int],
    max_queries: int,
    min_history: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Leave-last-out per user. Memory-efficient via pandas + chunked aggregation.

    For each eligible user (>= min_history+1 catalog-resolved events):
      input  = mean of all-but-last item embeddings, L2-normalized
      target = item_id of last event

    Eligible users are randomly sampled (seeded) down to `max_queries` to bound
    eval cost. The full 11.8M-interaction set easily exceeds 4 GB if dict-listed.

    v0 query construction; replaced in M3 by the real SID-LLM input.
    """
    print(f"  loading interactions via pyarrow ...")
    df = pq.read_table(str(interactions_path), columns=["user_id", "parent_asin", "timestamp_ms"]).to_pandas()
    print(f"  {len(df):,} raw interactions, {df['user_id'].nunique():,} unique users")

    # Drop rows for items NOT in our catalog.
    df = df[df["parent_asin"].isin(asin_to_iid.keys())].copy()
    df["item_id"] = df["parent_asin"].map(asin_to_iid).astype("int64")
    print(f"  {len(df):,} interactions after catalog filter")

    # Sort by (user_id, timestamp) so the last row per user_id is the held-out target.
    df = df.sort_values(["user_id", "timestamp_ms"], kind="mergesort")

    # Drop users with too few interactions (need at least min_history+1: history + target).
    counts = df.groupby("user_id", sort=False).size()
    eligible_users = counts[counts >= min_history + 1].index
    df = df[df["user_id"].isin(eligible_users)].copy()
    print(f"  {len(eligible_users):,} eligible users (>= {min_history + 1} events)")

    # Random sample of users for tractable eval.
    rng = np.random.default_rng(seed)
    if len(eligible_users) > max_queries:
        sampled_users = rng.choice(eligible_users.values, size=max_queries, replace=False)
        df = df[df["user_id"].isin(sampled_users)].copy()
        print(f"  sampled down to {max_queries:,} users for eval")

    # For each user: target = last row, history_iids = all but last.
    df = df.sort_values(["user_id", "timestamp_ms"], kind="mergesort")
    df["row_in_user"] = df.groupby("user_id", sort=False).cumcount()
    user_lengths = df.groupby("user_id", sort=False).size().rename("n_events")
    df = df.join(user_lengths, on="user_id")
    df["is_target"] = df["row_in_user"] == (df["n_events"] - 1)

    # Translate item_id -> embedding row index in catalog_emb_arr; drop unresolved.
    df["emb_idx"] = df["item_id"].map(iid_to_emb_idx)
    df = df[df["emb_idx"].notna()].copy()
    df["emb_idx"] = df["emb_idx"].astype("int64")

    # Aggregate per user.
    print(f"  aggregating mean history embeddings ...")
    queries: list[np.ndarray] = []
    targets: list[int] = []
    for uid, group in df.groupby("user_id", sort=False):
        target_row = group[group["is_target"]]
        history_rows = group[~group["is_target"]]
        if len(target_row) != 1 or len(history_rows) == 0:
            continue
        target_iid = int(target_row["item_id"].iloc[0])
        history_embs = catalog_emb_arr[history_rows["emb_idx"].values]
        q = history_embs.mean(axis=0)
        q = q / (np.linalg.norm(q) + 1e-9)
        queries.append(q.astype(np.float32))
        targets.append(target_iid)

    if not queries:
        return np.zeros((0, 0), dtype=np.float32), []
    print(f"  {len(queries):,} queries built")
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
@click.option(
    "--max-queries", default=20000, type=int,
    help="Cap on eval queries (random sample of users). v0 default keeps eval tractable.",
)
@click.option(
    "--min-history", default=2, type=int,
    help="Minimum prior items required per user (target is held out separately).",
)
@click.option("--seed", default=42, type=int)
def main(
    catalog_in: Path, embeddings_in: Path, interactions_in: Path,
    out: Path, ks: str, baseline_name: str,
    max_queries: int, min_history: int, seed: int,
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
    iid_to_emb_idx = {int(iid): i for i, iid in enumerate(item_ids)}

    print("Building user-history queries ...")
    queries, targets = _build_user_history_queries(
        interactions_in, asin_to_iid, emb_arr, iid_to_emb_idx,
        max_queries=max_queries, min_history=min_history, seed=seed,
    )

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
