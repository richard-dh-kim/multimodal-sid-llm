"""Build the CPT corpus: 50/50 mix of metadata and behavior sequences.

Metadata sequence (one per item with a SID):
  input:  "<seq> title: TITLE | category: SUB_CAT | description: TEXT[:500]"
  target: "<sid_T0><sid_T1><sid_T2><sid_T3><sid_eos>"

Behavior sequence (one per user with at least 2 catalog-resolved events):
  input:  "<seq> <sid_a0>...<sid_a3><sid_b0>...<sid_b3>...<sid_y0>...<sid_y3>"
  target: "<sid_z0><sid_z1><sid_z2><sid_z3><sid_eos>"
  where the user's last item is held out as the target.

Capped to ~150k metadata + ~150k behavior rows to keep training tractable.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import click
import pyarrow as pa
import pyarrow.parquet as pq

from sid_llm.data.text_clean import clean_text


DESC_MAX_CHARS = 500
DEFAULT_USER_HISTORY_CAP = 50  # clip user histories to last N events
DEFAULT_BEHAVIOR_USERS = 150_000


def build_metadata_sequence(item: dict) -> tuple[str, str]:
    """(input_text, target_text) for one catalog item with a SID."""
    title = (item.get("title") or "").strip()
    sub_cat = (item.get("sub_category") or "").strip() or "unknown"
    desc = clean_text(item.get("description"), max_chars=DESC_MAX_CHARS)
    parts = [f"<seq> title: {title}"]
    if sub_cat and sub_cat != "unknown":
        parts.append(f"category: {sub_cat}")
    if desc:
        parts.append(f"description: {desc}")
    inp = " | ".join(parts)
    sid_tokens = "".join(
        f"<sid_{int(item[f'sid_{i}'])}>" for i in range(4)
    )
    tgt = sid_tokens + "<sid_eos>"
    return inp, tgt


def build_behavior_sequence(
    sid_chain: list[tuple[int, int, int, int]]
) -> tuple[str, str] | None:
    """Convert a user's time-ordered SID chain into (history-input, last-item-target).

    Returns None if the chain has fewer than 2 items (no held-out target possible).
    """
    if len(sid_chain) < 2:
        return None
    history = sid_chain[:-1]
    last = sid_chain[-1]
    history_tokens = "".join(
        "".join(f"<sid_{c}>" for c in tup) for tup in history
    )
    inp = f"<seq> {history_tokens}"
    tgt = "".join(f"<sid_{c}>" for c in last) + "<sid_eos>"
    return inp, tgt


def aggregate_user_events(
    interactions: Iterable[dict],
    asin_to_sid: dict[str, tuple[int, int, int, int]],
    cap: int = DEFAULT_USER_HISTORY_CAP,
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Group interactions by user, sort chronologically, resolve SIDs, cap length."""
    by_user: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = defaultdict(list)
    for r in interactions:
        sid = asin_to_sid.get(r["parent_asin"])
        if sid is None:
            continue
        by_user[r["user_id"]].append((int(r["timestamp_ms"]), sid))

    out: dict[str, list[tuple[int, int, int, int]]] = {}
    for uid, events in by_user.items():
        events.sort(key=lambda x: x[0])
        if len(events) > cap:
            events = events[-cap:]
        out[uid] = [sid for _, sid in events]
    return out


@click.command()
@click.option(
    "--catalog-in", default="data/catalog/catalog_with_sid.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--interactions-in", default="data/catalog/interactions.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out", default="data/catalog/cpt_corpus.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--max-behavior-users", default=DEFAULT_BEHAVIOR_USERS, type=int)
@click.option("--user-history-cap", default=DEFAULT_USER_HISTORY_CAP, type=int)
@click.option("--seed", default=42, type=int)
def main(
    catalog_in: Path, interactions_in: Path, out: Path,
    max_behavior_users: int, user_history_cap: int, seed: int,
):
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print(f"Loading {catalog_in} ...")
    catalog = pq.read_table(str(catalog_in))
    items = catalog.to_pylist()
    items_with_sid = [
        it for it in items
        if all(it.get(f"sid_{i}") is not None for i in range(4))
    ]
    print(f"  catalog={catalog.num_rows:,}, with SIDs={len(items_with_sid):,}")

    print("Building metadata sequences ...")
    meta_rows = []
    for it in items_with_sid:
        inp, tgt = build_metadata_sequence(it)
        meta_rows.append({"seq_type": "metadata", "input_text": inp, "target_text": tgt})
    print(f"  {len(meta_rows):,} metadata rows")

    print("Building behavior sequences ...")
    asin_to_sid: dict[str, tuple[int, int, int, int]] = {}
    for it in items_with_sid:
        asin_to_sid[it["parent_asin"]] = tuple(int(it[f"sid_{i}"]) for i in range(4))

    # Stream batches; full interactions table doesn't fit in 16GB.
    pf = pq.ParquetFile(str(interactions_in))
    total_rows = pf.metadata.num_rows
    print(f"  {total_rows:,} raw interactions (streaming in batches)")

    by_user: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = defaultdict(list)
    seen = 0
    for batch in pf.iter_batches(
        batch_size=500_000,
        columns=["user_id", "parent_asin", "timestamp_ms"],
    ):
        d = batch.to_pydict()
        for uid, asin, ts in zip(d["user_id"], d["parent_asin"], d["timestamp_ms"]):
            sid = asin_to_sid.get(asin)
            if sid is None:
                continue
            by_user[uid].append((int(ts), sid))
        seen += batch.num_rows
        print(f"    streamed {seen:,}/{total_rows:,}")

    user_chains: dict[str, list[tuple[int, int, int, int]]] = {}
    for uid, events in by_user.items():
        events.sort(key=lambda x: x[0])
        if len(events) > user_history_cap:
            events = events[-user_history_cap:]
        user_chains[uid] = [sid for _, sid in events]
    del by_user
    print(f"  {len(user_chains):,} users with at least one catalog-resolved event")

    eligible = [uid for uid, chain in user_chains.items() if len(chain) >= 2]
    print(f"  {len(eligible):,} users with >= 2 events")
    if len(eligible) > max_behavior_users:
        eligible = rng.sample(eligible, max_behavior_users)

    behav_rows = []
    for uid in eligible:
        result = build_behavior_sequence(user_chains[uid])
        if result is None:
            continue
        inp, tgt = result
        behav_rows.append({"seq_type": "behavior", "input_text": inp, "target_text": tgt})
    print(f"  {len(behav_rows):,} behavior rows")

    all_rows = meta_rows + behav_rows
    rng.shuffle(all_rows)
    print(f"\nTotal corpus: {len(all_rows):,} rows ({len(meta_rows):,} metadata + {len(behav_rows):,} behavior)")

    table = pa.Table.from_pylist(all_rows)
    pq.write_table(table, str(out), compression="snappy")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {size_mb:.1f} MB to {out}")


if __name__ == "__main__":
    main()
