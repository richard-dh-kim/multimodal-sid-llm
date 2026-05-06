"""TIGER replication baseline (B3) trainer.

Vanilla TIGER recipe over the SID corpus. Mirrors `train_cpt.py`'s memory
hygiene (lazy PyArrow rows, top-level pickleable collator, HF safetensors save
with `max_shard_size`, `enable_checkpointing=False` on the Lightning Trainer)
so the same code path runs cleanly on both Linux GPU and a 16GB-RAM Windows
box. See `src/sid_llm/models/tiger_baseline.py` for the architectural choices.
"""
from __future__ import annotations

from pathlib import Path

import click
import lightning as L
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from sid_llm.models.tiger_baseline import build_tiger_baseline


class TigerBehaviorDataset(Dataset):
    """Reads cpt_corpus.parquet and yields (input_text, target_text) for behavior rows only.

    Filters lazily via PyArrow's `filter` kernel — does NOT materialize the table
    into a Python list, so 300k-row corpora stay cheap on a 16GB box. Holds the
    PyArrow columns directly in the same shape as `train_cpt.CPTSeqDataset`.
    """

    def __init__(self, corpus_path: Path):
        table = pq.read_table(str(corpus_path))
        if "seq_type" in table.column_names:
            mask = pc.equal(table.column("seq_type"), "behavior")
            table = table.filter(mask)
        self.table = table
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
    """Top-level callable so multi-worker DataLoader can pickle it on Windows.

    Shape-identical to `train_cpt.T5Collator`; duplicated here intentionally so
    the baseline can be deleted as a unit without touching the main trainer.
    """

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


class TigerLightning(L.LightningModule):
    """Small T5 from scratch + plain AdamW + OneCycleLR. No anchored optimizer,
    no soft prompts, no multimodal tower."""

    def __init__(
        self,
        tokenizer_path: Path,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_frac: float = 0.3,
        total_steps: int | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer, self.model = build_tiger_baseline(Path(tokenizer_path))
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
        # Plain AdamW — no AdamWAnchored. Betas (0.9, 0.98) match T5 / TIGER
        # convention (a slightly higher beta2 than torch's 0.999 default; see
        # T5 paper Appx D).
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.98),
        )
        if self.hparams.total_steps:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.hparams.total_steps,
                pct_start=self.hparams.warmup_frac,
                div_factor=8.0,
                final_div_factor=100.0,
                anneal_strategy="cos",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return optimizer


class HFSavePerEpoch(Callback):
    """Save the underlying T5 + tokenizer at the end of every train epoch
    in HF safetensors format. Mirrors the M3.6 callback so `sid_llm_eval`
    can load this baseline through the same `--ckpt-dir` path."""

    def __init__(self, ckpt_dir: Path):
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self._best_loss: float = float("inf")

    def on_train_epoch_end(self, trainer, pl_module):
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
    "--tokenizer-path", default="checkpoints/sid_llm/init/hf_model",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory holding the M3.5 expanded T5 tokenizer (vocab=33,128).",
)
@click.option(
    "--ckpt-dir", default="checkpoints/tiger_b3",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--epochs", default=8, type=int)
@click.option("--max-steps", default=-1, type=int, help="If >0, stop after this many train steps.")
@click.option("--batch-size", default=32, type=int,
              help="Per-device micro-batch. Effective batch = batch_size * accumulate_grad_batches.")
@click.option("--accumulate-grad-batches", default=4, type=int,
              help="Number of micro-batches per optimizer step.")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=True,
              help="Trade compute for activation memory. The TIGER-sized model fits "
                   "easily on 16GB even without it; flag is here for parity with the main trainer.")
@click.option("--lr", default=1e-3, type=float)
@click.option("--weight-decay", default=0.01, type=float)
@click.option("--warmup-frac", default=0.3, type=float,
              help="OneCycleLR pct_start: fraction of total steps spent ramping up.")
@click.option("--num-workers", default=0, type=int,
              help="Keep 0 on Windows: DataLoader workers there are spawned (no CoW), "
                   "so each one re-loads the corpus and inflates CPU RAM.")
@click.option("--smoke-cap", default=0, type=int, help="If >0, cap corpus to N rows (smoke).")
@click.option("--val-frac", default=0.01, type=float)
@click.option("--max-input-len", default=512, type=int)
@click.option("--max-target-len", default=16, type=int)
@click.option("--precision", default="bf16-mixed", type=str)
@click.option("--accelerator", default="auto", type=str)
def main(
    corpus_in, tokenizer_path, ckpt_dir, epochs, max_steps, batch_size,
    accumulate_grad_batches, gradient_checkpointing,
    lr, weight_decay, warmup_frac, num_workers, smoke_cap, val_frac,
    max_input_len, max_target_len, precision, accelerator,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {corpus_in} (behavior rows only) ...")
    full = TigerBehaviorDataset(corpus_in)
    n = len(full)
    print(f"  {n:,} behavior rows")
    if smoke_cap > 0:
        full = Subset(full, list(range(min(smoke_cap, n))))
        n = len(full)
        print(f"  SMOKE: capped to {n:,}")

    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    train_idxs = list(range(n_train))
    val_idxs = list(range(n_train, n))
    train_ds = Subset(full, train_idxs)
    val_ds = Subset(full, val_idxs)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    print(f"Building TIGER baseline (random init, tokenizer from {tokenizer_path}) ...")
    model = TigerLightning(
        tokenizer_path=tokenizer_path,
        lr=lr,
        weight_decay=weight_decay,
        warmup_frac=warmup_frac,
        gradient_checkpointing=gradient_checkpointing,
    )
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  TIGER baseline params: {n_params/1e6:.2f}M  (vocab={len(model.tokenizer):,})")

    collate = T5Collator(model.tokenizer, max_input_len=max_input_len, max_target_len=max_target_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate, pin_memory=True, persistent_workers=(num_workers > 0),
    )

    micro_batches_per_epoch = max(1, len(train_loader))
    opt_steps_per_epoch = max(1, micro_batches_per_epoch // max(1, accumulate_grad_batches))
    total_steps = max_steps if max_steps > 0 else opt_steps_per_epoch * epochs
    model.hparams.total_steps = total_steps

    callbacks = [HFSavePerEpoch(ckpt_dir)]
    logger = CSVLogger(save_dir="logs", name="tiger_b3")

    trainer = L.Trainer(
        max_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        enable_progress_bar=False,
        # Same rationale as train_cpt.py: HFSavePerEpoch handles persistence
        # via safetensors; Lightning's torch.save path trips on a Windows
        # bf16 zip-alignment bug for some model sizes.
        enable_checkpointing=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("\nTIGER baseline (B3) training done.")


if __name__ == "__main__":
    main()
