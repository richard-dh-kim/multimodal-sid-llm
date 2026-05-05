"""PyTorch Dataset for VL-CLIP fine-tuning."""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

from sid_llm.data.download_images import image_path_for_item
from sid_llm.data.text_clean import clean_text


class VLClipItemDataset(Dataset):
    """Reads catalog.parquet, filters to (split, has_image) rows whose JPGs
    are actually on disk under images_dir. __getitem__ returns a dict with
    'image' (PIL.Image), 'text' (cleaned title), 'item_id', 'sub_category'.

    Tokenization + image preprocessing happens in the collate_fn (so the dataset
    stays cheap and the heavy lifting is batched).
    """

    def __init__(
        self,
        catalog_path: Path,
        images_dir: Path,
        split: str = "train",
        text_max_chars: int = 200,
    ):
        self.images_dir = Path(images_dir)
        self.text_max_chars = text_max_chars

        cols = ["item_id", "title", "split", "has_image", "sub_category"]
        t = pq.read_table(str(catalog_path), columns=cols)
        df = t.to_pandas()

        # Filter to (split, has_image=True, image actually on disk).
        df = df[(df["split"] == split) & (df["has_image"])].copy()
        df = df[df["item_id"].apply(
            lambda iid: image_path_for_item(self.images_dir, int(iid)).exists()
        )].reset_index(drop=True)

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        iid = int(row["item_id"])
        ipath = image_path_for_item(self.images_dir, iid)
        try:
            image = Image.open(ipath)
            image.load()  # force decode now to surface OSError here, not later
            image = image.convert("RGB")
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            # Broken JPEGs (partial download, HTML error page saved as .jpg, etc.)
            # affect <0.1% of items in practice. Substitute a 224x224 gray placeholder
            # so the batch keeps its size and training continues. Logged once per worker.
            if not getattr(self, "_warned_broken", False):
                print(f"[VLClipItemDataset] WARN: broken image at {ipath}: {e}. "
                      f"Substituting gray placeholder. Further warnings suppressed.")
                self._warned_broken = True
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        text = clean_text(row["title"], max_chars=self.text_max_chars)
        return {
            "image": image,
            "text": text,
            "item_id": iid,
            "sub_category": str(row["sub_category"] or ""),
        }


class CLIPCollator:
    """Top-level callable so DataLoader workers can pickle it on Windows.

    A local closure (the previous `def collate(...)` inside `make_collate_fn`)
    cannot be pickled by Windows' spawn multiprocessing, which is what
    Lightning's DataLoader needs for num_workers>0. A class with `__call__`
    serializes cleanly.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [s["image"] for s in batch]
        texts = [s["text"] for s in batch]
        item_ids = [s["item_id"] for s in batch]
        encoded = self.processor(
            images=images, text=texts, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        )
        encoded["item_ids"] = item_ids
        return encoded


def make_collate_fn(processor):
    """Returns a picklable CLIPCollator instance.

    Kept as a factory function for API compatibility; previously returned a
    local closure that couldn't be pickled to DataLoader worker processes.
    """
    return CLIPCollator(processor)
