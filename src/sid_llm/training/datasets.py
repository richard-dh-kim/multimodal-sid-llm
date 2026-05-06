"""Datasets and collators for VL-CLIP and generative-retrieval fine-tuning."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from sid_llm.data.download_images import image_path_for_item
from sid_llm.data.text_clean import clean_text


class VLClipItemDataset(Dataset):
    """Catalog rows for VL-CLIP fine-tune. Filters to (split, has_image=True, JPG on disk)."""

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
            image.load()
            image = image.convert("RGB")
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            # Broken JPEG (partial download, HTML error saved as .jpg). Substitute a
            # gray placeholder so the batch keeps its size; warn once per worker.
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
    """Top-level callable so multi-worker DataLoader can pickle it on Windows."""

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
    return CLIPCollator(processor)


def _format_sid_target(sid_0: int, sid_1: int, sid_2: int, sid_3: int) -> str:
    """Render 4 codebook codes as the SID target string (matches build_cpt_corpus)."""
    return f"<sid_{int(sid_0)}><sid_{int(sid_1)}><sid_{int(sid_2)}><sid_{int(sid_3)}><sid_eos>"


class RetrievalDataset(Dataset):
    """Stratified 50/50 sequence-mode + search-mode retrieval examples.

    Sequence-mode: behavior rows from cpt_corpus.parquet (history-of-SIDs ->
    next-SID), fed to T5 via input_ids.
    Search-mode: CLIP item embedding -> own-SID. The LightningModule builds
    a 5-token encoder input from the embedding (4 soft tokens + <search>).

    Even indices are sequence rows, odd indices are search rows. Pair with
    `RetrievalBatchSampler` so each batch is uniform-mode. `__len__` is
    `2 * min(n_seq, n_search)`.
    """

    def __init__(
        self,
        corpus_path: Path,
        embeddings_path: Path | None = None,
        catalog_with_sid_path: Path | None = None,
        seed: int = 42,
        smoke_cap: int = 0,
    ):
        self.corpus_path = Path(corpus_path)
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.catalog_with_sid_path = (
            Path(catalog_with_sid_path) if catalog_with_sid_path else None
        )
        self._rng = np.random.default_rng(seed)

        table = pq.read_table(str(self.corpus_path))
        seq_types = table.column("seq_type").to_pylist()
        inputs = table.column("input_text").to_pylist()
        targets = table.column("target_text").to_pylist()

        seq_inputs: list[str] = []
        seq_targets: list[str] = []
        for s, i, t in zip(seq_types, inputs, targets):
            if s == "behavior":
                seq_inputs.append(i)
                seq_targets.append(t)

        self._seq_inputs = seq_inputs
        self._seq_targets = seq_targets

        if self.embeddings_path is None:
            raise ValueError(
                "embeddings_path is required for search-mode training. "
                "Pass --embeddings-path data/catalog/embeddings_b2.parquet."
            )
        if self.catalog_with_sid_path is None:
            raise ValueError(
                "catalog_with_sid_path is required for search-mode training. "
                "Pass --catalog-with-sid-path data/catalog/catalog_with_sid.parquet."
            )

        emb_table = pq.read_table(str(self.embeddings_path))
        emb_item_ids = emb_table.column("item_id").to_pylist()
        emb_vectors = np.stack(
            [np.asarray(v, dtype=np.float32) for v in emb_table.column("embedding").to_pylist()]
        )

        sid_table = pq.read_table(
            str(self.catalog_with_sid_path),
            columns=["item_id", "sid_0", "sid_1", "sid_2", "sid_3"],
        )
        sid_df = sid_table.to_pandas()
        sid_df = sid_df.dropna(subset=["sid_0", "sid_1", "sid_2", "sid_3"])
        sid_lookup: dict[int, str] = {
            int(row["item_id"]): _format_sid_target(
                row["sid_0"], row["sid_1"], row["sid_2"], row["sid_3"]
            )
            for _, row in sid_df.iterrows()
        }

        kept_embeddings: list[np.ndarray] = []
        kept_targets: list[str] = []
        for iid, vec in zip(emb_item_ids, emb_vectors):
            tgt = sid_lookup.get(int(iid))
            if tgt is None:
                continue
            kept_embeddings.append(vec)
            kept_targets.append(tgt)

        if not kept_targets:
            raise ValueError(
                "No (item_id, SID) pairs survived the join between embeddings "
                "and catalog_with_sid. Check that the two files cover overlapping items."
            )

        self._search_embeddings = np.stack(kept_embeddings)
        self._search_targets = kept_targets

        n_seq = len(self._seq_inputs)
        n_search = len(self._search_targets)
        n_each = min(n_seq, n_search)
        if smoke_cap > 0:
            n_each = min(n_each, max(1, smoke_cap // 2))

        self._seq_idx = self._rng.permutation(n_seq)[:n_each].tolist()
        self._search_idx = self._rng.permutation(n_search)[:n_each].tolist()
        self._n_each = n_each

    @property
    def search_mode(self) -> str:
        return "embedding"

    def __len__(self) -> int:
        return 2 * self._n_each

    def __getitem__(self, idx: int) -> dict:
        mode_is_seq = (idx % 2) == 0
        slot = idx // 2
        if mode_is_seq:
            j = self._seq_idx[slot]
            return {
                "mode": "sequence",
                "input_text": self._seq_inputs[j],
                "target_text": self._seq_targets[j],
            }
        j = self._search_idx[slot]
        return {
            "mode": "search",
            "query_embedding": torch.from_numpy(self._search_embeddings[j]),
            "target_text": self._search_targets[j],
        }


class RetrievalBatchSampler:
    """Alternating uniform-mode batches: even batch = sequence, odd = search.

    Pairs with RetrievalDataset's even=seq / odd=search index convention.
    """

    def __init__(
        self,
        n_each: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.n_each = n_each
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            per_mode = self.n_each // self.batch_size
        else:
            per_mode = (self.n_each + self.batch_size - 1) // self.batch_size
        return 2 * per_mode

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        seq_slots = np.arange(self.n_each)
        search_slots = np.arange(self.n_each)
        if self.shuffle:
            rng.shuffle(seq_slots)
            rng.shuffle(search_slots)

        seq_indices = (seq_slots * 2).tolist()
        search_indices = (search_slots * 2 + 1).tolist()

        bs = self.batch_size
        if self.drop_last:
            n_pairs = self.n_each // bs
        else:
            n_pairs = (self.n_each + bs - 1) // bs

        for b in range(n_pairs):
            yield seq_indices[b * bs : (b + 1) * bs]
            yield search_indices[b * bs : (b + 1) * bs]


class RetrievalCollator:
    """Top-level callable so multi-worker DataLoader can pickle it on Windows.

    Sequence batch -> {input_ids, attention_mask, labels}.
    Search batch -> {query_embeddings, labels}.
    Pad positions in labels are set to -100.
    """

    def __init__(self, tokenizer, max_input_len: int = 512, max_target_len: int = 16):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __call__(self, batch):
        modes = {b["mode"] for b in batch}
        if len(modes) != 1:
            raise ValueError(
                f"RetrievalCollator received a batch with mixed modes {modes}. "
                "Pair the dataset with RetrievalBatchSampler so each batch is "
                "uniform-mode."
            )
        mode = next(iter(modes))

        targets = [b["target_text"] for b in batch]
        tgt = self.tokenizer(
            targets, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_target_len,
        )
        labels = tgt["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        if mode == "sequence":
            inputs = [b["input_text"] for b in batch]
            enc = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_input_len,
            )
            return {
                "mode": "sequence",
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }
        emb = torch.stack([b["query_embedding"] for b in batch])
        return {
            "mode": "search",
            "query_embeddings": emb,
            "labels": labels,
        }
