"""Grounding DINO product-region cropper, VL-CLIP style.

Loads `IDEA-Research/grounding-dino-tiny` (~170M params). For each image,
runs detection with the product type as the text prompt; if the top box
clears a confidence threshold, crops to that box. Otherwise returns the
full image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_THRESHOLD = 0.35


@dataclass
class Crop:
    image: Image.Image
    cropped: bool
    score: float
    box: tuple[int, int, int, int] | None  # (x0, y0, x1, y1) or None if not cropped


def crop_with_threshold(
    image: Image.Image,
    score: float,
    threshold: float,
    box: tuple[float, float, float, float],
) -> Crop:
    if score < threshold:
        return Crop(image=image, cropped=False, score=score, box=None)
    w, h = image.size
    x0, y0, x1, y1 = box
    x0 = int(max(0, min(w, x0)))
    y0 = int(max(0, min(h, y0)))
    x1 = int(max(0, min(w, x1)))
    y1 = int(max(0, min(h, y1)))
    if x1 <= x0 or y1 <= y0:
        return Crop(image=image, cropped=False, score=score, box=None)
    return Crop(
        image=image.crop((x0, y0, x1, y1)),
        cropped=True,
        score=score,
        box=(x0, y0, x1, y1),
    )


class GroundingDinoCropper:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def crop_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[Crop]:
        """One Grounding-DINO inference per (image, prompt) pair, batched."""
        # Grounding DINO expects prompts as text strings ending in '.'
        formatted = [p if p.endswith(".") else p + "." for p in prompts]
        inputs = self.processor(
            images=images, text=formatted, return_tensors="pt", padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([img.size[::-1] for img in images]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.05,  # post-process keeps low-conf boxes; we filter later
            text_threshold=0.05,
            target_sizes=target_sizes,
        )

        crops: list[Crop] = []
        for img, res in zip(images, results):
            scores = res["scores"]
            boxes = res["boxes"]
            if len(scores) == 0:
                crops.append(Crop(image=img, cropped=False, score=0.0, box=None))
                continue
            top = torch.argmax(scores).item()
            crops.append(
                crop_with_threshold(
                    img, float(scores[top]), self.threshold, tuple(boxes[top].tolist())
                )
            )
        return crops
