"""Sequence-mode eval (history-of-SIDs -> next-SID) for a SID-LLM checkpoint.

Query format mirrors `build_cpt_corpus.build_behavior_sequence` so the
distribution matches sequence-mode training.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from sid_llm.eval.metrics import (
    hallucination_rate, ndcg_at_k, recall_at_k, silent_miss_rate,
)
from sid_llm.inference.beam_search import load_retriever


def _history_to_query_text(history_sids: list[tuple[int, int, int, int]]) -> str:
    """Mirror of build_cpt_corpus.build_behavior_sequence's input format."""
    history_tokens = "".join(
        "".join(f"<sid_{c}>" for c in tup) for tup in history_sids
    )
    return f"<seq> {history_tokens}"


def _build_seq_queries(
    interactions_df: pd.DataFrame,
    asin_to_iid: dict[str, int],
    iid_to_sid: dict[int, tuple[int, int, int, int]],
    max_queries: int,
    min_history: int,
    max_history_len: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    """Build (query_text, target_item_id) pairs for sequence-mode eval.

    Eligible users have >= min_history + 1 catalog-resolved interactions; the
    last item is the target, the preceding items (clipped to max_history_len)
    form the history.
    """
    df = interactions_df[interactions_df["parent_asin"].isin(asin_to_iid.keys())].copy()
    df["item_id"] = df["parent_asin"].map(asin_to_iid).astype("int64")
    df = df[df["item_id"].isin(iid_to_sid.keys())].copy()
    df = df.sort_values(["user_id", "timestamp_ms"], kind="mergesort")

    counts = df.groupby("user_id", sort=False).size()
    eligible = counts[counts >= min_history + 1].index
    df = df[df["user_id"].isin(eligible)].copy()

    rng = np.random.default_rng(seed)
    if len(eligible) > max_queries:
        sampled = rng.choice(eligible.values, size=max_queries, replace=False)
        df = df[df["user_id"].isin(sampled)].copy()
        df = df.sort_values(["user_id", "timestamp_ms"], kind="mergesort")

    queries: list[str] = []
    targets: list[int] = []
    for _, group in df.groupby("user_id", sort=False):
        iids = group["item_id"].tolist()
        if len(iids) < min_history + 1:
            continue
        target_iid = int(iids[-1])
        history_iids = iids[:-1]
        if len(history_iids) > max_history_len:
            history_iids = history_iids[-max_history_len:]
        history_sids = [iid_to_sid[int(i)] for i in history_iids if int(i) in iid_to_sid]
        if len(history_sids) < min_history:
            continue
        queries.append(_history_to_query_text(history_sids))
        targets.append(target_iid)
    return queries, targets


def _load_interactions(path: Path) -> pd.DataFrame:
    return pq.read_table(
        str(path),
        columns=["user_id", "parent_asin", "timestamp_ms"],
    ).to_pandas()


@click.command()
@click.option("--ckpt-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Retrieval-fine-tune output dir (contains the HF model).")
@click.option("--catalog-in", default="data/catalog/catalog_with_sid.parquet",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--interactions-in", default="data/catalog/interactions.parquet",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sid-to-item", default="data/catalog/sid_to_item.pkl",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sid-trie", default="data/catalog/sid_trie.pkl",
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", default="eval/results/sid_llm_seq.json",
              type=click.Path(dir_okay=False, path_type=Path))
@click.option("--ks", default="5,10,50", type=str)
@click.option("--max-queries", default=2_000, type=int,
              help="Beam search is slow; cap users for tractable eval.")
@click.option("--min-history", default=2, type=int)
@click.option("--max-history-len", default=50, type=int,
              help="Clip history to the last N items (mirrors M3.6 cap).")
@click.option("--num-beams", default=10, type=int)
@click.option("--constrained/--unconstrained", default=True)
@click.option("--baseline-name", default="SID_LLM_seq", type=str)
@click.option("--seed", default=42, type=int)
@click.option("--device", default=None, type=str)
def main(
    ckpt_dir, catalog_in, interactions_in, sid_to_item, sid_trie, out,
    ks, max_queries, min_history, max_history_len, num_beams, constrained,
    baseline_name, seed, device,
):
    out.parent.mkdir(parents=True, exist_ok=True)

    k_list = [int(x) for x in ks.split(",")]
    max_k = max(k_list)
    if num_beams < max_k:
        print(
            f"WARNING: num_beams ({num_beams}) < max(ks) ({max_k}); "
            f"auto-bumping num_beams to {max_k}."
        )
        num_beams = max_k

    print(f"Loading retriever from {ckpt_dir} ...")
    retr = load_retriever(ckpt_dir, sid_to_item, sid_trie, device=device)
    print(f"  device={retr.device}")

    print(f"Loading catalog from {catalog_in} ...")
    cat = pq.read_table(str(catalog_in))
    asin_to_iid = dict(zip(
        cat.column("parent_asin").to_pylist(),
        cat.column("item_id").to_pylist(),
    ))
    iid_to_sid: dict[int, tuple[int, int, int, int]] = {}
    for iid, s0, s1, s2, s3 in zip(
        cat.column("item_id").to_pylist(),
        cat.column("sid_0").to_pylist(), cat.column("sid_1").to_pylist(),
        cat.column("sid_2").to_pylist(), cat.column("sid_3").to_pylist(),
    ):
        if s0 is not None:
            iid_to_sid[int(iid)] = (int(s0), int(s1), int(s2), int(s3))
    print(f"  {len(iid_to_sid):,} items have SIDs")

    print(f"Loading interactions from {interactions_in} ...")
    interactions_df = _load_interactions(interactions_in)
    print(f"  {len(interactions_df):,} raw interactions")

    print("Building sequence queries ...")
    queries, targets = _build_seq_queries(
        interactions_df, asin_to_iid, iid_to_sid,
        max_queries=max_queries, min_history=min_history,
        max_history_len=max_history_len, seed=seed,
    )
    print(f"  {len(queries):,} queries built")
    if not queries:
        print("WARNING: no queries built; aborting.")
        return

    print(
        f"Running beam search over {len(queries):,} queries "
        f"(num_beams={num_beams}, constrained={constrained}) ..."
    )
    preds: list[list[int]] = []
    top1_sids: list[tuple[int, ...]] = []
    for q in tqdm(queries, desc="retrieve"):
        item_ids, sids = retr.retrieve_from_text(
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

    results = {f"recall@{k}": recall_at_k(preds, targets, k) for k in k_list}
    results.update({f"ndcg@{k}": ndcg_at_k(preds, targets, k) for k in k_list})
    results["hallucination_rate"] = halluc
    results["silent_miss_rate"] = silent

    print(f"\n{baseline_name} results (sequence mode, next-item):")
    for kk, vv in results.items():
        print(f"  {kk}: {vv:.4f}")

    out.write_text(json.dumps({
        "baseline": baseline_name,
        "task": "user-history-to-next-item-sid",
        "n_queries": len(queries),
        "num_beams": num_beams,
        "constrained": constrained,
        "max_history_len": max_history_len,
        "metrics": results,
    }, indent=2))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
