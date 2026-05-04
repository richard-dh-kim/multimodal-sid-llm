"""Build the canonical catalog.parquet by joining items + interactions + image presence.

Adds: item_id (sequential), image_local_path, has_image, split (train/val/test).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from sid_llm.data.download_images import image_path_for_item


@dataclass
class SplitConfig:
    test_window_days: int = 7
    val_window_days: int = 7


def add_item_ids(items: pa.Table) -> pa.Table:
    n = items.num_rows
    return items.append_column("item_id", pa.array(list(range(n)), type=pa.int64()))


def add_image_paths(items: pa.Table, images_dir: Path) -> pa.Table:
    item_ids = items.column("item_id").to_pylist()
    paths: list[str] = []
    has: list[bool] = []
    for iid in item_ids:
        p = image_path_for_item(images_dir, int(iid))
        if p.exists() and p.stat().st_size > 100:
            paths.append(str(p))
            has.append(True)
        else:
            paths.append("")
            has.append(False)
    items = items.append_column("image_local_path", pa.array(paths, type=pa.string()))
    items = items.append_column("has_image", pa.array(has, type=pa.bool_()))
    return items


def _maybe_sort_by_item_id(table: pa.Table) -> pa.Table:
    if "item_id" in table.column_names:
        return table.sort_by("item_id")
    return table


def add_temporal_splits(
    items: pa.Table, interactions: pa.Table, cfg: SplitConfig
) -> pa.Table:
    """For each item, compute its 'last interaction timestamp' across all users.
    Items split by their last-interaction date relative to the global max timestamp.
    Requires `items` to have a 'parent_asin' string column.
    Returns a table sorted by item_id (if present) so row N has item_id=N.
    """
    if "parent_asin" not in items.column_names:
        raise ValueError("items table must have a 'parent_asin' column")

    if interactions.num_rows == 0:
        # No interactions → all train
        return _maybe_sort_by_item_id(
            items.append_column(
                "split", pa.array(["train"] * items.num_rows, type=pa.string())
            )
        )

    asins = interactions.column("parent_asin")
    ts = interactions.column("timestamp_ms")
    df = pa.Table.from_arrays([asins, ts], names=["parent_asin", "timestamp_ms"])
    # group_by max
    last_ts = df.group_by("parent_asin").aggregate([("timestamp_ms", "max")])
    # join onto items (NOTE: pyarrow joins do not preserve row order; we sort below)
    joined = items.join(last_ts, keys="parent_asin", join_type="left outer")

    last_ts_col = joined.column("timestamp_ms_max").to_pylist()
    if not last_ts_col:
        return _maybe_sort_by_item_id(
            joined.append_column(
                "split", pa.array(["train"] * joined.num_rows, type=pa.string())
            )
        )

    valid_ts = [t for t in last_ts_col if t is not None and t > 0]
    if not valid_ts:
        splits = ["train"] * len(last_ts_col)
    else:
        global_max = max(valid_ts)
        ms_per_day = 24 * 60 * 60 * 1000
        test_cutoff = global_max - cfg.test_window_days * ms_per_day
        val_cutoff = test_cutoff - cfg.val_window_days * ms_per_day
        splits: list[str] = []
        for t in last_ts_col:
            if t is None or t <= 0:
                splits.append("train")
            elif t > test_cutoff:
                splits.append("test")
            elif t > val_cutoff:
                splits.append("val")
            else:
                splits.append("train")

    return _maybe_sort_by_item_id(
        joined.append_column("split", pa.array(splits, type=pa.string()))
    )


@click.command()
@click.option(
    "--items-in", default="data/catalog/items.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--interactions-in", default="data/catalog/interactions.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--images-dir", default="data/images",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--out", default="data/catalog/catalog.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
def main(items_in: Path, interactions_in: Path, images_dir: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading items + interactions ...")
    items = pq.read_table(str(items_in))
    interactions = pq.read_table(str(interactions_in))

    print(f"  {items.num_rows:,} items, {interactions.num_rows:,} interactions")

    items = add_item_ids(items)
    items = add_image_paths(items, images_dir)
    items = add_temporal_splits(items, interactions, SplitConfig())

    print(f"\nSplit counts:")
    splits = items.column("split").to_pylist()
    for s in ("train", "val", "test"):
        print(f"  {s}: {sum(1 for x in splits if x == s):,}")
    has_image = items.column("has_image").to_pylist()
    print(f"\nImage availability: {sum(has_image):,} / {len(has_image):,} = {sum(has_image)/len(has_image):.1%}")

    pq.write_table(items, str(out), compression="snappy")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {size_mb:.1f} MB to {out}.")


if __name__ == "__main__":
    main()
