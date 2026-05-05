"""CLI: fine-tune VL-CLIP with symmetric InfoNCE on our catalog."""
from __future__ import annotations

from pathlib import Path

import click
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from sid_llm.models.vl_clip import VLClipLightning
from sid_llm.training.datasets import VLClipItemDataset, make_collate_fn


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
    "--ckpt-dir", default="checkpoints/vl_clip",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--model-id", default="openai/clip-vit-base-patch32", type=str)
@click.option("--epochs", default=6, type=int)
@click.option("--batch-size", default=128, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--weight-decay", default=0.01, type=float)
@click.option("--warmup-ratio", default=0.05, type=float)
@click.option("--num-workers", default=4, type=int)
@click.option("--patience", default=2, type=int)
@click.option("--smoke-cap", default=0, type=int, help="If >0, cap each split to N samples (smoke test)")
@click.option("--precision", default="bf16-mixed", type=str)
def main(
    catalog_in, images_dir, ckpt_dir, model_id, epochs, batch_size,
    lr, weight_decay, warmup_ratio, num_workers, patience, smoke_cap, precision,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading datasets from {catalog_in} ...")
    train_ds = VLClipItemDataset(catalog_in, images_dir, split="train")
    val_ds = VLClipItemDataset(catalog_in, images_dir, split="val")
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    if smoke_cap > 0:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(smoke_cap, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(smoke_cap, len(val_ds)))))
        print(f"  SMOKE: train={len(train_ds)}  val={len(val_ds)}")

    # Instantiate model FIRST so we can use its processor in collate_fn.
    print(f"Instantiating VL-CLIP from {model_id} ...")
    model = VLClipLightning(model_id=model_id, lr=lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio)
    collate = make_collate_fn(model.processor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # Total steps (for schedule).
    steps_per_epoch = max(1, len(train_loader))
    model.hparams.total_steps = steps_per_epoch * epochs

    callbacks = [
        EarlyStopping(monitor="val_recall_at_10", mode="max", patience=patience, verbose=True),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="vl_clip-epoch{epoch:02d}-r10{val_recall_at_10:.4f}",
            monitor="val_recall_at_10", mode="max", save_top_k=1, save_last=True,
            auto_insert_metric_name=False,
        ),
    ]
    logger = CSVLogger(save_dir="logs", name="vl_clip")

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        val_check_interval=1.0,
        enable_progress_bar=False,  # rich progress bar can hang on redirected stdout
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save the final HF-format weights for downstream use by embed_items.py
    final_dir = ckpt_dir / "final_hf"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(final_dir))
    model.processor.save_pretrained(str(final_dir))
    print(f"\nSaved fine-tuned weights to {final_dir}")
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        print(f"Best Lightning ckpt: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
