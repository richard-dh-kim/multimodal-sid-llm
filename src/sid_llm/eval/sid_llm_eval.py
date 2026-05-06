"""Evaluate a SID-LLM checkpoint via beam search retrieval against test interactions."""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from sid_llm.eval.metrics import (
    hallucination_rate, ndcg_at_k, recall_at_k, silent_miss_rate,
)
from sid_llm.inference.beam_search import load_retriever


def _build_user_history_queries_text(
    interactions_path: Path,
    asin_to_iid: dict[str, int],
    iid_to_sid: dict[int, tuple[int, int, int, int]],
    catalog_titles: dict[int, str],
    max_queries: int,
    min_history: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    """Title-of-last-history-item -> next-item id. Heuristic text-mode eval."""
    df = pq.read_table(
        str(interactions_path),
        columns=["user_id", "parent_asin", "timestamp_ms"],
    ).to_pandas()
    df = df[df["parent_asin"].isin(asin_to_iid.keys())].copy()
    df["item_id"] = df["parent_asin"].map(asin_to_iid).astype("int64")
    df = df.sort_values(["user_id", "timestamp_ms"], kind="mergesort")

    counts = df.groupby("user_id", sort=False).size()
    eligible = counts[counts >= min_history + 1].index
    df = df[df["user_id"].isin(eligible)].copy()

    rng = np.random.default_rng(seed)
    if len(eligible) > max_queries:
        sampled = rng.choice(eligible.values, size=max_queries, replace=False)
        df = df[df["user_id"].isin(sampled)].copy()

    df = df.sort_values(["user_id", "timestamp_ms"], kind="mergesort")
    df["row_in_user"] = df.groupby("user_id", sort=False).cumcount()
    user_lengths = df.groupby("user_id", sort=False).size().rename("n_events")
    df = df.join(user_lengths, on="user_id")
    df["is_target"] = df["row_in_user"] == (df["n_events"] - 1)
    df["is_last_history"] = df["row_in_user"] == (df["n_events"] - 2)

    queries: list[str] = []
    targets: list[int] = []
    for _, group in df.groupby("user_id", sort=False):
        target_row = group[group["is_target"]]
        history_last = group[group["is_last_history"]]
        if len(target_row) != 1 or len(history_last) != 1:
            continue
        target_iid = int(target_row["item_id"].iloc[0])
        history_iid = int(history_last["item_id"].iloc[0])
        title = catalog_titles.get(history_iid, "")
        if not title:
            continue
        queries.append(title)
        targets.append(target_iid)
    return queries, targets


@click.command()
@click.option("--ckpt-dir", default="checkpoints/sid_llm/init/hf_model",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--catalog-in", default="data/catalog/catalog_with_sid.parquet",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--interactions-in", default="data/catalog/interactions.parquet",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sid-to-item", default="data/catalog/sid_to_item.pkl",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sid-trie", default="data/catalog/sid_trie.pkl",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", default="eval/results/sid_llm.json",
              type=click.Path(dir_okay=False, path_type=Path))
@click.option("--ks", default="10,50,100", type=str)
@click.option("--max-queries", default=2_000, type=int,
              help="Beam search is slow vs MIPS; default cap is 2000.")
@click.option("--min-history", default=2, type=int)
@click.option("--num-beams", default=10, type=int)
@click.option("--constrained/--unconstrained", default=True)
@click.option("--baseline-name", default="SID_LLM_constrained", type=str)
@click.option("--seed", default=42, type=int)
@click.option("--device", default=None, type=str)
def main(
    ckpt_dir, catalog_in, interactions_in, sid_to_item, sid_trie, out,
    ks, max_queries, min_history, num_beams, constrained, baseline_name, seed, device,
):
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading retriever from {ckpt_dir} ...")
    retr = load_retriever(ckpt_dir, sid_to_item, sid_trie, device=device)
    print(f"  device={retr.device}")

    print(f"Loading catalog ...")
    cat = pq.read_table(str(catalog_in))
    asin_to_iid = dict(zip(
        cat.column("parent_asin").to_pylist(),
        cat.column("item_id").to_pylist(),
    ))
    titles = dict(zip(
        cat.column("item_id").to_pylist(),
        cat.column("title").to_pylist(),
    ))
    iid_to_sid: dict[int, tuple[int, int, int, int]] = {}
    for iid, s0, s1, s2, s3 in zip(
        cat.column("item_id").to_pylist(),
        cat.column("sid_0").to_pylist(), cat.column("sid_1").to_pylist(),
        cat.column("sid_2").to_pylist(), cat.column("sid_3").to_pylist(),
    ):
        if s0 is not None:
            iid_to_sid[int(iid)] = (int(s0), int(s1), int(s2), int(s3))

    print(f"Building queries ...")
    queries, targets = _build_user_history_queries_text(
        interactions_in, asin_to_iid, iid_to_sid, titles,
        max_queries=max_queries, min_history=min_history, seed=seed,
    )
    print(f"  {len(queries):,} queries built")
    if not queries:
        print("WARNING: no queries built; aborting.")
        return

    k_list = [int(x) for x in ks.split(",")]
    max_k = max(k_list)
    if num_beams < max_k:
        print(
            f"WARNING: num_beams ({num_beams}) < max(ks) ({max_k}); "
            f"auto-bumping num_beams to {max_k}."
        )
        num_beams = max_k

    print(f"Running beam search over {len(queries):,} queries (num_beams={num_beams}, constrained={constrained}) ...")
    preds: list[list[int]] = []
    all_sids: list[tuple[int, int, int, int]] = []
    for q in tqdm(queries, desc="retrieve"):
        item_ids, sids = retr.retrieve_from_text(q, k=max_k, num_beams=num_beams, constrained=constrained)
        # Push silent-miss (-1) to the back so it doesn't displace hits in recall@k.
        preds.append([iid for iid in item_ids if iid >= 0] + [iid for iid in item_ids if iid < 0])
        all_sids.extend(sids[:1])

    valid_sids = set(retr.sid_to_item.keys())
    halluc = hallucination_rate(all_sids, valid_sids)
    silent = silent_miss_rate(all_sids, retr.sid_to_item)

    results = {f"recall@{k}": recall_at_k(preds, targets, k) for k in k_list}
    results.update({f"ndcg@{k}": ndcg_at_k(preds, targets, k) for k in k_list})
    results["hallucination_rate"] = halluc
    results["silent_miss_rate"] = silent

    print(f"\n{baseline_name} results:")
    for kk, vv in results.items():
        print(f"  {kk}: {vv:.4f}")

    out.write_text(json.dumps({
        "baseline": baseline_name,
        "n_queries": len(queries),
        "num_beams": num_beams,
        "constrained": constrained,
        "metrics": results,
    }, indent=2))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
