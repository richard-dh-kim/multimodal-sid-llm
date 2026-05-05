from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
from sid_llm.training.datasets import VLClipItemDataset


def test_dataset_filters_to_train_split_with_images(tmp_path: Path):
    catalog = pa.Table.from_pylist([
        {"item_id": 0, "title": "Drill", "split": "train", "has_image": True, "sub_category": "Power & Hand Tools"},
        {"item_id": 1, "title": "Saw", "split": "test", "has_image": True, "sub_category": "Power & Hand Tools"},
        {"item_id": 2, "title": "Hammer", "split": "train", "has_image": False, "sub_category": "Power & Hand Tools"},
        {"item_id": 3, "title": "Wrench", "split": "train", "has_image": True, "sub_category": "Hardware"},
    ])
    catalog_path = tmp_path / "catalog.parquet"
    pq.write_table(catalog, str(catalog_path))

    images_dir = tmp_path / "images"
    # Create dummy images for items 0 and 3 (in correct shard dirs)
    for iid in [0, 3]:
        shard = f"{iid:03d}" if iid < 1000 else f"{iid // 1000:03d}"
        d = images_dir / shard
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (224, 224), "red").save(d / f"{iid}.jpg")

    ds = VLClipItemDataset(catalog_path, images_dir, split="train")
    # Should include item 0 (train + has_image + image-on-disk)
    # Should include item 3 (train + has_image + image-on-disk)
    # Should exclude item 1 (test split)
    # Should exclude item 2 (has_image=False)
    assert len(ds) == 2


def test_dataset_returns_tensor_and_text(tmp_path: Path):
    catalog = pa.Table.from_pylist([
        {"item_id": 0, "title": "A drill", "split": "train", "has_image": True, "sub_category": "X"},
    ])
    catalog_path = tmp_path / "catalog.parquet"
    pq.write_table(catalog, str(catalog_path))

    images_dir = tmp_path / "images"
    (images_dir / "000").mkdir(parents=True)
    Image.new("RGB", (224, 224), "blue").save(images_dir / "000" / "0.jpg")

    ds = VLClipItemDataset(catalog_path, images_dir, split="train")
    sample = ds[0]
    assert "image" in sample and isinstance(sample["image"], Image.Image)
    assert "text" in sample and sample["text"]
    assert "item_id" in sample
    assert "sub_category" in sample
