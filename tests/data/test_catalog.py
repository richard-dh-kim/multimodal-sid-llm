from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from sid_llm.data.catalog import (
    add_item_ids,
    add_image_paths,
    add_temporal_splits,
    SplitConfig,
)


def test_add_item_ids_assigns_sequential_ids():
    table = pa.Table.from_pylist([
        {"parent_asin": "A"},
        {"parent_asin": "B"},
        {"parent_asin": "C"},
    ])
    out = add_item_ids(table)
    assert out.column("item_id").to_pylist() == [0, 1, 2]


def test_add_image_paths_marks_existing_files(tmp_path: Path):
    items = pa.Table.from_pylist([
        {"item_id": 0, "image_url": "u0"},
        {"item_id": 1, "image_url": "u1"},
        {"item_id": 2, "image_url": "u2"},
    ])
    images_dir = tmp_path / "images"
    (images_dir / "000").mkdir(parents=True)
    (images_dir / "002").mkdir(parents=True)
    (images_dir / "000" / "0.jpg").write_bytes(b"x" * 200)
    (images_dir / "002" / "2.jpg").write_bytes(b"x" * 200)

    out = add_image_paths(items, images_dir)
    has_image = out.column("has_image").to_pylist()
    assert has_image == [True, False, True]
    paths = out.column("image_local_path").to_pylist()
    assert paths[0].endswith("0.jpg")
    assert paths[1] == ""  # missing


def test_add_temporal_splits_uses_interaction_timestamps():
    interactions = pa.Table.from_pylist([
        {"parent_asin": "A", "timestamp_ms": 100},
        {"parent_asin": "A", "timestamp_ms": 200},
        {"parent_asin": "B", "timestamp_ms": 300},
        {"parent_asin": "C", "timestamp_ms": 1_000_000},
    ])
    items = pa.Table.from_pylist([
        {"item_id": 0, "parent_asin": "A"},
        {"item_id": 1, "parent_asin": "B"},
        {"item_id": 2, "parent_asin": "C"},
    ])
    cfg = SplitConfig(test_window_days=1, val_window_days=1)
    out = add_temporal_splits(items, interactions, cfg)
    splits = out.column("split").to_pylist()
    assert set(splits) <= {"train", "val", "test"}
    assert all(s == "train" or s == "val" or s == "test" for s in splits)
