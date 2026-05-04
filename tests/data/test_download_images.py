from pathlib import Path
import pytest
from sid_llm.data.download_images import (
    image_path_for_item,
    derive_item_id_from_row_index,
    is_already_downloaded,
)


def test_image_path_for_item_uses_subdir_sharding(tmp_path: Path):
    p = image_path_for_item(tmp_path, item_id=12345)
    assert p.name == "12345.jpg"
    # shard prefix prevents 150k files in one directory
    assert p.parent.name == "012"  # first 3 digits of zero-padded item_id


def test_image_path_for_item_handles_small_ids(tmp_path: Path):
    p = image_path_for_item(tmp_path, item_id=0)
    assert p.name == "0.jpg"
    assert p.parent.name == "000"


def test_derive_item_id_returns_zero_indexed_row_position():
    assert derive_item_id_from_row_index(0) == 0
    assert derive_item_id_from_row_index(149_999) == 149_999


def test_is_already_downloaded_returns_true_when_file_exists(tmp_path: Path):
    f = tmp_path / "010" / "10.jpg"
    f.parent.mkdir()
    f.write_bytes(b"jpeg")
    assert is_already_downloaded(tmp_path, 10) is True


def test_is_already_downloaded_returns_false_when_missing(tmp_path: Path):
    assert is_already_downloaded(tmp_path, 10) is False


def test_is_already_downloaded_returns_false_when_zero_bytes(tmp_path: Path):
    f = tmp_path / "010" / "10.jpg"
    f.parent.mkdir()
    f.write_bytes(b"")
    assert is_already_downloaded(tmp_path, 10) is False
