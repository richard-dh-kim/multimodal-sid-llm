from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
from sid_llm.data.build_sid_catalog import (
    build_sid_to_item,
    join_catalog_with_sids,
    detect_collisions,
)


def test_build_sid_to_item_picks_lowest_item_id_on_collision():
    """Two items share SID (1,2,3,4); the dict should pick the lower item_id."""
    item2feat = pa.Table.from_pylist([
        {"item_id": 5, "sid_0": 1, "sid_1": 2, "sid_2": 3, "sid_3": 4},
        {"item_id": 3, "sid_0": 1, "sid_1": 2, "sid_2": 3, "sid_3": 4},
        {"item_id": 7, "sid_0": 9, "sid_1": 9, "sid_2": 9, "sid_3": 9},
    ])
    sid_to_item = build_sid_to_item(item2feat)
    assert sid_to_item[(1, 2, 3, 4)] == 3  # lowest of {5, 3}
    assert sid_to_item[(9, 9, 9, 9)] == 7
    assert len(sid_to_item) == 2  # one entry per unique SID


def test_detect_collisions_returns_groups_with_n_ge_2():
    item2feat = pa.Table.from_pylist([
        {"item_id": 5, "sid_0": 1, "sid_1": 2, "sid_2": 3, "sid_3": 4},
        {"item_id": 3, "sid_0": 1, "sid_1": 2, "sid_2": 3, "sid_3": 4},
        {"item_id": 7, "sid_0": 9, "sid_1": 9, "sid_2": 9, "sid_3": 9},  # singleton, NOT a collision
    ])
    coll = detect_collisions(item2feat)
    # Only one collision group: SID (1,2,3,4) with items [3, 5]
    rows = coll.to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert (row["sid_0"], row["sid_1"], row["sid_2"], row["sid_3"]) == (1, 2, 3, 4)
    assert row["n_items"] == 2
    assert sorted(row["item_ids"]) == [3, 5]


def test_join_catalog_preserves_row_order_and_count():
    catalog = pa.Table.from_pylist([
        {"item_id": 0, "title": "A"},
        {"item_id": 1, "title": "B"},
        {"item_id": 2, "title": "C"},  # not in item2feat
    ])
    item2feat = pa.Table.from_pylist([
        {"item_id": 0, "sid_0": 10, "sid_1": 20, "sid_2": 30, "sid_3": 40},
        {"item_id": 1, "sid_0": 11, "sid_1": 21, "sid_2": 31, "sid_3": 41},
    ])
    out = join_catalog_with_sids(catalog, item2feat)
    assert out.num_rows == 3  # row count unchanged
    item_ids = out.column("item_id").to_pylist()
    assert item_ids == [0, 1, 2]  # order unchanged
    sid0 = out.column("sid_0").to_pylist()
    assert sid0 == [10, 11, None]  # null for item 2 which has no SID


def test_join_catalog_handles_empty_item2feat():
    catalog = pa.Table.from_pylist([{"item_id": 0, "title": "A"}])
    item2feat = pa.Table.from_pylist([], schema=pa.schema([
        ("item_id", pa.int64()),
        ("sid_0", pa.int64()),
        ("sid_1", pa.int64()),
        ("sid_2", pa.int64()),
        ("sid_3", pa.int64()),
    ]))
    out = join_catalog_with_sids(catalog, item2feat)
    assert out.num_rows == 1
    assert out.column("sid_0").to_pylist() == [None]
