"""Build the SID-augmented catalog and reverse-lookup artifacts.

Given:
  - catalog.parquet         (one row per item, no SIDs)
  - item2feat.parquet       (one row per quantized item: item_id + sid_0..sid_3)

Produces:
  - catalog_with_sid.parquet   (catalog left-joined with SIDs; same row count)
  - sid_to_item.pkl            (dict[tuple, int]; on collision picks lowest item_id)
  - sid_collisions.parquet     (collisions log; one row per group with n_items >= 2)
  - sid_trie.pkl               (SIDTrie built over all unique SIDs, for inference)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq

from sid_llm.inference.trie import SIDTrie

CODEBOOK_SIZE = 1024  # default; SIDTrie uses this as vocab_size


def build_sid_to_item(item2feat: pa.Table) -> dict[tuple[int, int, int, int], int]:
    """Return {sid_tuple: item_id}; on collision, picks the LOWEST item_id."""
    item_ids = item2feat.column("item_id").to_pylist()
    sids_0 = item2feat.column("sid_0").to_pylist()
    sids_1 = item2feat.column("sid_1").to_pylist()
    sids_2 = item2feat.column("sid_2").to_pylist()
    sids_3 = item2feat.column("sid_3").to_pylist()

    out: dict[tuple[int, int, int, int], int] = {}
    for iid, s0, s1, s2, s3 in zip(item_ids, sids_0, sids_1, sids_2, sids_3):
        key = (int(s0), int(s1), int(s2), int(s3))
        prev = out.get(key)
        if prev is None or int(iid) < prev:
            out[key] = int(iid)
    return out


def detect_collisions(item2feat: pa.Table) -> pa.Table:
    """Return a Parquet table with one row per non-trivial collision group.
    Columns: sid_0, sid_1, sid_2, sid_3, n_items, item_ids (list of int).
    Sorted by n_items descending so the worst collisions are first.
    """
    item_ids = item2feat.column("item_id").to_pylist()
    sids_0 = item2feat.column("sid_0").to_pylist()
    sids_1 = item2feat.column("sid_1").to_pylist()
    sids_2 = item2feat.column("sid_2").to_pylist()
    sids_3 = item2feat.column("sid_3").to_pylist()

    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for iid, s0, s1, s2, s3 in zip(item_ids, sids_0, sids_1, sids_2, sids_3):
        key = (int(s0), int(s1), int(s2), int(s3))
        groups.setdefault(key, []).append(int(iid))

    rows = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        rows.append({
            "sid_0": key[0], "sid_1": key[1], "sid_2": key[2], "sid_3": key[3],
            "n_items": len(members),
            "item_ids": sorted(members),
        })
    rows.sort(key=lambda r: -r["n_items"])
    if not rows:
        # Empty schema-correct table for downstream consumers.
        return pa.Table.from_pylist([], schema=pa.schema([
            ("sid_0", pa.int64()), ("sid_1", pa.int64()),
            ("sid_2", pa.int64()), ("sid_3", pa.int64()),
            ("n_items", pa.int64()),
            ("item_ids", pa.list_(pa.int64())),
        ]))
    return pa.Table.from_pylist(rows)


def join_catalog_with_sids(catalog: pa.Table, item2feat: pa.Table) -> pa.Table:
    """Append sid_0..sid_3 columns to catalog (left join on item_id).
    Items in catalog but missing from item2feat get NULL SIDs.
    Row order and count preserved.
    """
    # Use a dict-based join to avoid pyarrow.Table.join's list-column quirk.
    cat_iids = catalog.column("item_id").to_pylist()
    iid_to_sid: dict[int, tuple[int, int, int, int]] = {}
    for iid, s0, s1, s2, s3 in zip(
        item2feat.column("item_id").to_pylist(),
        item2feat.column("sid_0").to_pylist(),
        item2feat.column("sid_1").to_pylist(),
        item2feat.column("sid_2").to_pylist(),
        item2feat.column("sid_3").to_pylist(),
    ):
        iid_to_sid[int(iid)] = (int(s0), int(s1), int(s2), int(s3))

    cols = {0: [], 1: [], 2: [], 3: []}
    for iid in cat_iids:
        sid = iid_to_sid.get(int(iid))
        for k in range(4):
            cols[k].append(sid[k] if sid is not None else None)

    out = catalog
    for k in range(4):
        out = out.append_column(f"sid_{k}", pa.array(cols[k], type=pa.int64()))
    return out


@click.command()
@click.option(
    "--catalog-in", default="data/catalog/catalog.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--item2feat-in", default="data/catalog/item2feat.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--catalog-out", default="data/catalog/catalog_with_sid.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--sid-to-item-out", default="data/catalog/sid_to_item.pkl",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--collisions-out", default="data/catalog/sid_collisions.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--trie-out", default="data/catalog/sid_trie.pkl",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--codebook-size", default=CODEBOOK_SIZE, type=int)
def main(
    catalog_in: Path, item2feat_in: Path,
    catalog_out: Path, sid_to_item_out: Path, collisions_out: Path, trie_out: Path,
    codebook_size: int,
) -> None:
    catalog_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {catalog_in} and {item2feat_in} ...")
    catalog = pq.read_table(str(catalog_in))
    item2feat = pq.read_table(str(item2feat_in))
    print(f"  catalog={catalog.num_rows:,}  item2feat={item2feat.num_rows:,}")

    print("Joining catalog with SIDs ...")
    catalog_with_sid = join_catalog_with_sids(catalog, item2feat)
    pq.write_table(catalog_with_sid, str(catalog_out), compression="snappy")
    print(f"  -> {catalog_out}  ({catalog_out.stat().st_size / 1024 / 1024:.1f} MB)")

    print("Building sid_to_item dict ...")
    sid_to_item = build_sid_to_item(item2feat)
    with open(sid_to_item_out, "wb") as f:
        pickle.dump(sid_to_item, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {len(sid_to_item):,} unique SIDs  -> {sid_to_item_out}")

    print("Detecting collisions ...")
    collisions = detect_collisions(item2feat)
    pq.write_table(collisions, str(collisions_out), compression="snappy")
    n_coll = collisions.num_rows
    n_total = item2feat.num_rows
    n_unique = len(sid_to_item)
    n_lost = n_total - n_unique
    print(f"  {n_coll:,} collision groups; {n_lost:,} items lost to dedup ({n_lost/n_total:.2%})")
    if n_coll > 0:
        # Show worst 5 collisions.
        head = collisions.slice(0, min(5, n_coll))
        for row in head.to_pylist():
            print(f"    sid={(row['sid_0'], row['sid_1'], row['sid_2'], row['sid_3'])} "
                  f"n_items={row['n_items']} ids={row['item_ids'][:5]}{'...' if row['n_items'] > 5 else ''}")

    print("Building SIDTrie ...")
    valid_sids = list(sid_to_item.keys())
    trie = SIDTrie(valid_sids, vocab_size=codebook_size)
    with open(trie_out, "wb") as f:
        pickle.dump(trie, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {len(valid_sids):,} SIDs in trie  -> {trie_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
