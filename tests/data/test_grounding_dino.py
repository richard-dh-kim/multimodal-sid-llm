from pathlib import Path
from PIL import Image
import pytest
from sid_llm.data.grounding_dino import (
    Crop,
    crop_with_threshold,
)


def test_crop_with_threshold_returns_full_image_below_threshold(tmp_path):
    img = Image.new("RGB", (200, 200), "red")
    crop = crop_with_threshold(img, score=0.2, threshold=0.35, box=(0, 0, 50, 50))
    assert crop.cropped is False
    assert crop.image.size == (200, 200)


def test_crop_with_threshold_crops_above_threshold(tmp_path):
    img = Image.new("RGB", (200, 200), "red")
    crop = crop_with_threshold(img, score=0.5, threshold=0.35, box=(10, 10, 110, 110))
    assert crop.cropped is True
    assert crop.image.size == (100, 100)
    assert crop.score == 0.5


def test_crop_handles_box_outside_image_bounds():
    img = Image.new("RGB", (200, 200), "red")
    crop = crop_with_threshold(img, score=0.9, threshold=0.35, box=(-10, -10, 250, 250))
    # should clip to image bounds
    assert crop.image.size == (200, 200)
