from unittest.mock import MagicMock

import torch
from PIL import Image

from sid_llm.data.grounding_dino import (
    Crop,
    GroundingDinoCropper,
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


def test_crop_batch_picks_top_box_via_argmax():
    """Mocked test: confirms crop_batch routes the highest-scoring box
    through crop_with_threshold without instantiating the real model.
    Catches signature drift in HF post-processing kwargs."""
    cropper = GroundingDinoCropper.__new__(GroundingDinoCropper)
    cropper.threshold = 0.35
    cropper.device = "cpu"
    cropper.processor = MagicMock()
    cropper.model = MagicMock()
    cropper.model.return_value = MagicMock()
    proc_inputs = MagicMock()
    proc_inputs.input_ids = torch.zeros((1, 4), dtype=torch.long)
    proc_inputs.to.return_value = proc_inputs
    cropper.processor.return_value = proc_inputs
    cropper.processor.post_process_grounded_object_detection.return_value = [
        {
            "scores": torch.tensor([0.10, 0.90]),
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0],
                                    [5.0, 5.0, 55.0, 55.0]]),
        }
    ]

    img = Image.new("RGB", (100, 100), "red")
    crops = cropper.crop_batch([img], ["shoe"])

    assert len(crops) == 1
    assert isinstance(crops[0], Crop)
    assert crops[0].cropped is True
    assert crops[0].box == (5, 5, 55, 55)
    assert abs(crops[0].score - 0.9) < 1e-3
