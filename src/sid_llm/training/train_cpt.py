"""CPT training CLI: continued pre-training of the SID-vocab-expanded T5 on the mixed corpus."""
from __future__ import annotations

import math
from pathlib import Path

import click
import lightning as L
import pyarrow.parquet as pq
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast


class CPTSeqDataset(Dataset):
    """Reads cpt_corpus.parquet and yields (input_text, target_text) per row.

    Holds the PyArrow columns directly instead of materializing 300k Python
    dicts — keeps baseline CPU RAM low so the per-epoch HF save has headroom.
    """

    def __init__(self, corpus_path: Path):
        self.table = pq.read_table(str(corpus_path))
        self._input_col = self.table.column("input_text")
        self._target_col = self.table.column("target_text")

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_text": self._input_col[idx].as_py(),
            "target_text": self._target_col[idx].as_py(),
        }


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def nullcontext():
    return _NullContext()


class T5Collator:
    """Top-level callable so multi-worker DataLoader can pickle it on Windows."""

    def __init__(self, tokenizer, max_input_len: int = 512, max_target_len: int = 16):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __call__(self, batch):
        inputs = [b["input_text"] for b in batch]
        targets = [b["target_text"] for b in batch]
        enc = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_input_len,
        )
        ctx = (
            self.tokenizer.as_target_tokenizer()
            if hasattr(self.tokenizer, "as_target_tokenizer")
            else nullcontext()
        )
        with ctx:
            tgt = self.tokenizer(
                targets, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_target_len,
            )
        labels = tgt["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


class CPTLightning(L.LightningModule):
    """T5 prefix-LM CPT.

    Loads the M3.5 init checkpoint (T5-base with vocab expanded to 33,128 tokens).
    Standard T5 forward(labels=...) gives cross-entropy on target SIDs (and any text).
    """

    def __init__(
        self,
        init_dir: Path,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = T5TokenizerFast.from_pretrained(str(init_dir))
        self.model = T5ForConditionalGeneration.from_pretrained(str(init_dir))
        if gradient_checkpointing:
            # T5 requires use_cache=False to be compatible with grad checkpointing.
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        bs = batch["input_ids"].size(0)
        self.log("train_loss", out.loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        bs = batch["input_ids"].size(0)
        self.log("val_loss", out.loss, prog_bar=True, on_epoch=True, batch_size=bs)
        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.98),
        )
        if self.hparams.total_steps:
            warmup = self.hparams.warmup_steps

            def lr_lambda(step):
                if step < warmup:
                    return (step + 1) / max(1, warmup)
                progress = (step - warmup) / max(1, self.hparams.total_steps - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return optimizer


class HFSavePerEpoch(Callback):
    """Save the underlying T5 + tokenizer at the end of every train epoch
    in HF safetensors format. Avoids torch.save bf16 zip-alignment issues."""

    def __init__(self, ckpt_dir: Path):
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self._best_loss: float = float("inf")

    def on_train_epoch_end(self, trainer, pl_module):
        # max_shard_size caps peak bytes() per shard; otherwise safetensors._tobytes()
        # copies the full 894MB state in one alloc and OOMs on a 16GB Windows box.
        save_kwargs = {"max_shard_size": "200MB"}
        latest = self.ckpt_dir / "hf_latest"
        latest.mkdir(parents=True, exist_ok=True)
        pl_module.model.save_pretrained(str(latest), **save_kwargs)
        pl_module.tokenizer.save_pretrained(str(latest))
        cur = trainer.callback_metrics.get("val_loss")
        if cur is None:
            return
        cur_v = float(cur.detach().cpu()) if hasattr(cur, "detach") else float(cur)
        if cur_v < self._best_loss:
            self._best_loss = cur_v
            best = self.ckpt_dir / "hf_best"
            best.mkdir(parents=True, exist_ok=True)
            pl_module.model.save_pretrained(str(best), **save_kwargs)
            pl_module.tokenizer.save_pretrained(str(best))


@click.command()
@click.option(
    "--corpus-in", default="data/catalog/cpt_corpus.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--init-dir", default="checkpoints/sid_llm/init/hf_model",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--ckpt-dir", default="checkpoints/sid_llm/cpt",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--epochs", default=2, type=int)
@click.option("--max-steps", default=-1, type=int, help="If >0, stop after this many train steps.")
@click.option("--batch-size", default=8, type=int,
              help="Per-device micro-batch. Effective batch = batch_size * accumulate_grad_batches.")
@click.option("--accumulate-grad-batches", default=4, type=int,
              help="Number of micro-batches per optimizer step.")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=True,
              help="Trade compute for activation memory; recommended on 16GB GPUs.")
@click.option("--lr", default=3e-4, type=float)
@click.option("--weight-decay", default=0.01, type=float)
@click.option("--warmup-steps", default=500, type=int)
@click.option("--num-workers", default=0, type=int,
              help="Keep 0 on Windows: DataLoader workers there are spawned (no CoW), "
                   "so each one re-loads the corpus and inflates CPU RAM.")
@click.option("--smoke-cap", default=0, type=int, help="If >0, cap corpus to N rows (smoke).")
@click.option("--val-frac", default=0.01, type=float)
@click.option("--max-input-len", default=512, type=int)
@click.option("--max-target-len", default=16, type=int)
@click.option("--precision", default="bf16-mixed", type=str)
def main(
    corpus_in, init_dir, ckpt_dir, epochs, max_steps, batch_size,
    accumulate_grad_batches, gradient_checkpointing,
    lr, weight_decay, warmup_steps, num_workers, smoke_cap, val_frac,
    max_input_len, max_target_len, precision,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {corpus_in} ...")
    full = CPTSeqDataset(corpus_in)
    n = len(full)
    print(f"  {n:,} rows")
    if smoke_cap > 0:
        from torch.utils.data import Subset
        full = Subset(full, list(range(min(smoke_cap, n))))
        n = len(full)
        print(f"  SMOKE: capped to {n:,}")

    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    train_idxs = list(range(n_train))
    val_idxs = list(range(n_train, n))
    from torch.utils.data import Subset
    train_ds = Subset(full, train_idxs)
    val_ds = Subset(full, val_idxs)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    print(f"Instantiating SID-LLM (CPT) from {init_dir} ...")
    model = CPTLightning(init_dir=init_dir, lr=lr, weight_decay=weight_decay,
                          warmup_steps=warmup_steps,
                          gradient_checkpointing=gradient_checkpointing)
    collate = T5Collator(model.tokenizer, max_input_len=max_input_len, max_target_len=max_target_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate, pin_memory=True, persistent_workers=(num_workers > 0),
    )

    # Scheduler steps = optimizer steps, not micro-batches.
    micro_batches_per_epoch = max(1, len(train_loader))
    opt_steps_per_epoch = max(1, micro_batches_per_epoch // max(1, accumulate_grad_batches))
    total_steps = max_steps if max_steps > 0 else opt_steps_per_epoch * epochs
    model.hparams.total_steps = total_steps

    callbacks = [HFSavePerEpoch(ckpt_dir)]
    logger = CSVLogger(save_dir="logs", name="sid_llm_cpt")

    trainer = L.Trainer(
        max_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        accelerator="auto",
        devices=1,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        enable_progress_bar=False,
        # Disable Lightning's auto-checkpointing (torch.save's zipfile writer trips
        # over a Windows bf16 alignment bug for large models). HFSavePerEpoch
        # handles persistence via HF safetensors.
        enable_checkpointing=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("\nCPT training done.")


if __name__ == "__main__":
    main()
