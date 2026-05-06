"""Generative-retrieval fine-tune (M3.7).

Loads the CPT'd checkpoint (M3.6) and trains it on a stratified 50/50 mix of
sequence-mode and search-mode examples. Optionally uses AdamWAnchored to
L2-SP-regularize toward the CPT init weights so the fine-tune doesn't forget.

Search-mode forward pass (per the spec):
    soft   = W_q @ q + p_i           (4 virtual encoder tokens, [B, 4, d_model])
    search = embed(<search>)         ([B, 1, d_model])
    inputs_embeds = concat([soft, search], dim=1)   ([B, 5, d_model])
    out    = T5(inputs_embeds=..., attention_mask=ones(B,5), labels=...)

W_q, p_i are initialized from `checkpoints/sid_llm/init/soft_prompt.pt` (M3.5
output) when available. Sequence-mode batches go through the standard
`input_ids`/`attention_mask` path.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import click
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from sid_llm.training.datasets import (
    RetrievalBatchSampler,
    RetrievalCollator,
    RetrievalDataset,
)
from sid_llm.training.optimizers import AdamWAnchored


# Must match the constants in sid_llm.models.sid_llm. Hard-coding rather than
# importing to keep this file dependency-light at test time.
QUERY_EMBED_DIM = 512
NUM_SOFT_PROMPT_TOKENS = 4


class _SubsetWithLen(Subset):
    """Subset that exposes the underlying dataset's `_n_each` if present.

    RetrievalBatchSampler needs the number of (seq, search) pairs in a *split*,
    not the full dataset. We compute that by counting even/odd indices in the
    Subset's `indices` list.
    """

    @property
    def n_each(self) -> int:
        evens = sum(1 for i in self.indices if i % 2 == 0)
        odds = sum(1 for i in self.indices if i % 2 == 1)
        return min(evens, odds)


class RetrievalLightning(L.LightningModule):
    """T5 generative retrieval fine-tune with search-mode soft-prompt fusion.

    Owns the soft-prompt projection (W_q: 512->768) and learned per-position
    offsets (p_i: [4, 768]). Initializes them from `soft_prompt_path` when
    that file exists; otherwise warns and falls back to small random init.
    """

    def __init__(
        self,
        init_dir: Path,
        soft_prompt_path: Path | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.005,
        total_steps: int | None = None,
        pct_start: float = 0.4,
        gradient_checkpointing: bool = False,
        use_anchored: bool = True,
        new_init_noise: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = T5TokenizerFast.from_pretrained(str(init_dir))
        self.model = T5ForConditionalGeneration.from_pretrained(str(init_dir))
        if gradient_checkpointing:
            # T5 requires use_cache=False to be compatible with grad checkpointing.
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        d_model = self.model.config.d_model
        self.query_projection = nn.Linear(QUERY_EMBED_DIM, d_model, bias=True)
        self.soft_prompt_offsets = nn.Parameter(
            torch.randn(NUM_SOFT_PROMPT_TOKENS, d_model) * new_init_noise
        )

        # Resolve the <search> token id. The M3.5 init script adds <search>
        # to the tokenizer; if for any reason it's missing here we fall back
        # to <seq> with a loud warning so the run still completes.
        self._search_token_id = self.tokenizer.convert_tokens_to_ids("<search>")
        if self._search_token_id == self.tokenizer.unk_token_id:
            seq_id = self.tokenizer.convert_tokens_to_ids("<seq>")
            warnings.warn(
                "<search> token is not in the tokenizer vocab. Falling back "
                "to <seq>'s embedding for the mode marker. This is a "
                "regression — verify M3.5 init pipeline.",
                stacklevel=2,
            )
            self._search_token_id = seq_id

        if soft_prompt_path is not None:
            self._load_soft_prompt(Path(soft_prompt_path))

    # ------------------------------------------------------------------
    # Soft-prompt init/load helpers
    # ------------------------------------------------------------------
    def _load_soft_prompt(self, path: Path) -> None:
        """Load `query_projection` weight/bias and `soft_prompt_offsets` from
        the M3.5 init dict at `path`. Missing file -> warn and keep random init.

        The expected on-disk layout (matching `init_sid_llm.py`):
            {
              "query_projection.state_dict": OrderedDict({weight, bias}),
              "soft_prompt_offsets": Tensor[4, 768],
              ... (other metadata fields ignored)
            }
        """
        if not path.exists():
            warnings.warn(
                f"soft_prompt path {path} not found; using random init for "
                "query_projection and soft_prompt_offsets.",
                stacklevel=2,
            )
            return
        sd = torch.load(str(path), map_location="cpu", weights_only=False)
        qp = sd.get("query_projection.state_dict")
        if qp is not None:
            # Direct assignment so this works even if shapes shifted (e.g.
            # different d_model). Fail loudly on shape mismatch.
            self.query_projection.load_state_dict(qp)
        else:
            warnings.warn(
                f"{path} missing 'query_projection.state_dict' key; "
                "skipping query_projection init.",
                stacklevel=2,
            )
        offsets = sd.get("soft_prompt_offsets")
        if offsets is not None:
            with torch.no_grad():
                self.soft_prompt_offsets.copy_(offsets.to(self.soft_prompt_offsets.dtype))
        else:
            warnings.warn(
                f"{path} missing 'soft_prompt_offsets' key; "
                "keeping random init.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------
    def soft_prompt_from_query(self, query_embed: torch.Tensor) -> torch.Tensor:
        """[B, 512] -> [B, 4, d_model], identical math to SIDLLMLightning."""
        projected = self.query_projection(query_embed)        # [B, d_model]
        return projected.unsqueeze(1) + self.soft_prompt_offsets.unsqueeze(0)

    def _search_forward(self, query_embed: torch.Tensor, labels: torch.Tensor):
        """Build the 5-token encoder input and run the T5 forward."""
        b = query_embed.size(0)
        device = query_embed.device

        soft = self.soft_prompt_from_query(query_embed)       # [B, 4, d_model]

        search_ids = torch.full(
            (b, 1), fill_value=self._search_token_id, dtype=torch.long, device=device
        )
        search_emb = self.model.shared(search_ids)            # [B, 1, d_model]

        inputs_embeds = torch.cat([soft, search_emb], dim=1)  # [B, 5, d_model]
        attention_mask = torch.ones(
            (b, inputs_embeds.size(1)), dtype=torch.long, device=device
        )
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _step(self, batch, stage: str):
        mode = batch["mode"]
        if mode == "sequence":
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            bs = batch["input_ids"].size(0)
        elif mode == "search":
            out = self._search_forward(
                query_embed=batch["query_embeddings"],
                labels=batch["labels"],
            )
            bs = batch["query_embeddings"].size(0)
        else:
            raise ValueError(f"Unknown batch mode {mode!r}")

        # Per-mode loss logging is handy for diagnosing whether one path is
        # diverging while the other trains fine.
        log_kw = dict(on_step=(stage == "train"), on_epoch=True, batch_size=bs)
        self.log(f"{stage}_loss", out.loss, prog_bar=True, **log_kw)
        self.log(f"{stage}_loss_{mode}", out.loss, **log_kw)
        return out.loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # ALL trainable params, not just self.model.parameters() — soft-prompt
        # params live on `self`, not on `self.model`, so they would otherwise
        # be excluded from the optimizer.
        params = list(self.parameters())
        if self.hparams.use_anchored:
            optimizer = AdamWAnchored(
                params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.98),
            )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.98),
            )
        if self.hparams.total_steps:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.hparams.total_steps,
                pct_start=self.hparams.pct_start,
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
    in HF safetensors format. Mirrors the M3.6 callback so downstream tools
    (beam_search, eval) can reuse the same load path.

    Also writes `soft_prompt.pt` alongside the HF model so search-mode
    inference can reload the soft-prompt projection. Same key layout as the
    M3.5 init file.
    """

    def __init__(self, ckpt_dir: Path):
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self._best_loss: float = float("inf")

    def _save(self, target_dir: Path, pl_module) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        pl_module.model.save_pretrained(str(target_dir), max_shard_size="200MB")
        pl_module.tokenizer.save_pretrained(str(target_dir))
        torch.save(
            {
                "query_projection.state_dict": pl_module.query_projection.state_dict(),
                "soft_prompt_offsets": pl_module.soft_prompt_offsets.detach().cpu(),
            },
            str(target_dir / "soft_prompt.pt"),
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self._save(self.ckpt_dir / "hf_latest", pl_module)
        cur = trainer.callback_metrics.get("val_loss")
        if cur is None:
            return
        cur_v = float(cur.detach().cpu()) if hasattr(cur, "detach") else float(cur)
        if cur_v < self._best_loss:
            self._best_loss = cur_v
            self._save(self.ckpt_dir / "hf_best", pl_module)


@click.command()
@click.option(
    "--corpus-in", default="data/catalog/cpt_corpus.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--embeddings-path", default="data/catalog/embeddings_b2.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CLIP item embeddings (parquet with item_id, embedding columns) "
         "used as queries in search-mode training. Required.",
)
@click.option(
    "--catalog-with-sid-path", default="data/catalog/catalog_with_sid.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Catalog parquet with item_id + sid_0..sid_3 columns; used to look "
         "up each search-mode query's target SID.",
)
@click.option(
    "--init-dir", default="checkpoints/sid_llm/cpt/hf_best",
    type=click.Path(file_okay=False, path_type=Path),
    help="CPT'd checkpoint to fine-tune from. Default points at the M3.6 hf_best output.",
)
@click.option(
    "--soft-prompt-path", default="checkpoints/sid_llm/init/soft_prompt.pt",
    type=click.Path(dir_okay=False, path_type=Path),
    help="State-dict produced by M3.5 init for query_projection + "
         "soft_prompt_offsets. If missing, random init + warning.",
)
@click.option(
    "--ckpt-dir", default="checkpoints/sid_llm/retrieval",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--epochs", default=3, type=int)
@click.option("--max-steps", default=-1, type=int, help="If >0, stop after this many train steps.")
@click.option("--batch-size", default=8, type=int,
              help="Per-device micro-batch. Effective batch = batch_size * accumulate_grad_batches. "
                   "Plan calls for effective batch 32 (8 * 4 by default).")
@click.option("--accumulate-grad-batches", default=4, type=int,
              help="Number of micro-batches per optimizer step.")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=True,
              help="Trade compute for activation memory; recommended on 16GB GPUs.")
@click.option("--lr", default=1e-4, type=float)
@click.option("--weight-decay", default=0.005, type=float)
@click.option("--pct-start", default=0.4, type=float, help="OneCycleLR warmup fraction.")
@click.option("--use-anchored/--no-anchored", default=True,
              help="Use AdamWAnchored (L2-SP toward init_dir weights) instead of plain AdamW.")
@click.option("--num-workers", default=0, type=int,
              help="Keep 0 on Windows: DataLoader workers there are spawned (no CoW), "
                   "so each one re-loads the corpus and inflates CPU RAM.")
@click.option("--smoke-cap", default=0, type=int,
              help="If >0, cap dataset to ~N rows total (smoke).")
@click.option("--val-frac", default=0.01, type=float)
@click.option("--max-input-len", default=512, type=int)
@click.option("--max-target-len", default=16, type=int)
@click.option("--precision", default="bf16-mixed", type=str)
@click.option("--accelerator", default="auto", type=str,
              help="Set to 'cpu' for smoke verification while another job holds the GPU.")
@click.option("--seed", default=42, type=int)
def main(
    corpus_in, embeddings_path, catalog_with_sid_path, init_dir, soft_prompt_path,
    ckpt_dir, epochs, max_steps, batch_size,
    accumulate_grad_batches, gradient_checkpointing,
    lr, weight_decay, pct_start, use_anchored, num_workers, smoke_cap, val_frac,
    max_input_len, max_target_len, precision, accelerator, seed,
):
    L.seed_everything(seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading retrieval corpus from {corpus_in} ...")
    full = RetrievalDataset(
        corpus_path=corpus_in,
        embeddings_path=embeddings_path,
        catalog_with_sid_path=catalog_with_sid_path,
        seed=seed,
        smoke_cap=smoke_cap,
    )
    n = len(full)
    print(f"  {n:,} rows  (search_mode={full.search_mode})")

    n_val = max(2, int(n * val_frac))
    # Keep val/train disjoint AND mode-balanced: val is the last n_val rows
    # (which alternate seq/search, so the slice is balanced too).
    n_train = n - n_val
    train_idxs = list(range(n_train))
    val_idxs = list(range(n_train, n))
    train_ds = _SubsetWithLen(full, train_idxs)
    val_ds = _SubsetWithLen(full, val_idxs)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    print(f"Instantiating SID-LLM (retrieval) from {init_dir} ...")
    if not Path(init_dir).exists():
        raise click.ClickException(
            f"--init-dir {init_dir} does not exist. Run the M3.6 CPT first, or "
            f"point --init-dir at checkpoints/sid_llm/init/hf_model for a from-init smoke."
        )
    model = RetrievalLightning(
        init_dir=init_dir,
        soft_prompt_path=soft_prompt_path,
        lr=lr,
        weight_decay=weight_decay,
        pct_start=pct_start,
        gradient_checkpointing=gradient_checkpointing,
        use_anchored=use_anchored,
    )
    collate = RetrievalCollator(
        model.tokenizer, max_input_len=max_input_len, max_target_len=max_target_len
    )

    # RetrievalBatchSampler emits batches per (seq, search) pair, so the
    # micro-batch count per epoch is 2x the number of pairs.
    train_sampler = RetrievalBatchSampler(
        n_each=train_ds.n_each, batch_size=batch_size, shuffle=True, seed=seed,
        drop_last=True,
    )
    val_sampler = RetrievalBatchSampler(
        n_each=val_ds.n_each, batch_size=batch_size, shuffle=False, seed=seed,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=num_workers,
        collate_fn=collate, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=num_workers,
        collate_fn=collate, pin_memory=True, persistent_workers=(num_workers > 0),
    )

    micro_batches_per_epoch = max(1, len(train_sampler))
    opt_steps_per_epoch = max(1, micro_batches_per_epoch // max(1, accumulate_grad_batches))
    total_steps = max_steps if max_steps > 0 else opt_steps_per_epoch * epochs
    model.hparams.total_steps = total_steps

    callbacks = [HFSavePerEpoch(ckpt_dir)]
    logger = CSVLogger(save_dir="logs", name="sid_llm_retrieval")

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
        # Disable Lightning auto-checkpointing; HFSavePerEpoch handles persistence.
        enable_checkpointing=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("\nRetrieval fine-tune done.")


if __name__ == "__main__":
    main()
