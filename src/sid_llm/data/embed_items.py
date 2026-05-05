"""Compute fused multimodal embeddings for every item using a CLIP model.

For B1 baseline: vanilla CLIP (no fine-tuning).
For embeddings used in tokenizer training (M3+): pass --model-checkpoint <vl_clip_ckpt>
once VL-CLIP is fine-tuned in M2.

Reads raw images directly via image_path_for_item — Grounding DINO cropping
is deferred (see spec FU-1). Sample inspection showed Amazon large-variant
images are already centered studio shots.
"""
from __future__ import annotations

import time
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from sid_llm.data.download_images import image_path_for_item
from sid_llm.data.text_clean import clean_text
from sid_llm.models.clip_embedder import ClipEmbedder


@click.command()
@click.option(
    "--catalog-in", default="data/catalog/catalog.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--images-dir", default="data/images",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--out", default="data/catalog/embeddings_b1.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--model-checkpoint", default="openai/clip-vit-base-patch32", type=str,
    help="HF model id OR local directory of fine-tuned weights.",
)
@click.option("--batch-size", default=64, type=int)
def main(
    catalog_in: Path, images_dir: Path, out: Path,
    model_checkpoint: str, batch_size: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    embedder = ClipEmbedder(model_id=model_checkpoint)
    print(f"device={embedder.device}  model={model_checkpoint}")

    catalog = pq.read_table(str(catalog_in))
    item_ids = catalog.column("item_id").to_pylist()
    titles = catalog.column("title").to_pylist()
    has_image = catalog.column("has_image").to_pylist()

    rows: list[dict] = []
    pbar = tqdm(total=len(item_ids), desc="Embedding", smoothing=0.05)
    start = time.time()
    failed = 0

    batch_imgs: list[Image.Image] = []
    batch_texts: list[str] = []
    batch_ids: list[int] = []

    def flush():
        if not batch_imgs:
            return
        embs = embedder.embed_image_text_batch(batch_imgs, batch_texts)
        for iid, e in zip(batch_ids, embs):
            rows.append({"item_id": int(iid), "embedding": e.tolist()})
        for img in batch_imgs:
            img.close()
        batch_imgs.clear()
        batch_texts.clear()
        batch_ids.clear()

    for i, iid in enumerate(item_ids):
        if not has_image[i]:
            pbar.update(1)
            continue
        ipath = image_path_for_item(images_dir, int(iid))
        if not ipath.exists():
            failed += 1
            pbar.update(1)
            continue
        try:
            img = Image.open(ipath).convert("RGB")
        except Exception:
            failed += 1
            pbar.update(1)
            continue
        text = clean_text(titles[i], max_chars=200)
        batch_imgs.append(img)
        batch_texts.append(text)
        batch_ids.append(int(iid))
        if len(batch_imgs) >= batch_size:
            flush()
        pbar.update(1)
    flush()
    pbar.close()

    elapsed = time.time() - start
    print(f"\nFinished {len(rows):,} embeddings in {elapsed/60:.1f} min  (failed_io={failed})")

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out), compression="snappy")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  → {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
