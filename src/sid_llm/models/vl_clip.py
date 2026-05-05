"""VL-CLIP Lightning module: fine-tunes CLIP ViT-B/32 with symmetric InfoNCE."""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from sid_llm.training.losses import symmetric_infonce


DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class VLClipLightning(L.LightningModule):
    """Lightning module wrapping HF CLIPModel with symmetric InfoNCE training.

    Tracks training loss and a small-scale val Recall@10 (computed against the
    val set's own item embeddings, NOT the full catalog -- kept cheap).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        total_steps: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self._val_image_emb_buf: list[torch.Tensor] = []
        self._val_text_emb_buf: list[torch.Tensor] = []

    def forward(self, **batch):
        return self.model(**batch)

    def _step_embeds(self, batch):
        # Strip non-tensor extras before forwarding.
        forward_inputs = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = self.model(**forward_inputs)
        image_embeds = outputs.image_embeds  # already L2-normed by HF CLIP
        text_embeds = outputs.text_embeds
        logit_scale = self.model.logit_scale
        return image_embeds, text_embeds, logit_scale

    def training_step(self, batch, batch_idx):
        img_emb, txt_emb, logit_scale = self._step_embeds(batch)
        loss = symmetric_infonce(img_emb, txt_emb, logit_scale)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/logit_scale", logit_scale.exp().detach(), on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_start(self):
        self._val_image_emb_buf = []
        self._val_text_emb_buf = []

    def validation_step(self, batch, batch_idx):
        img_emb, txt_emb, logit_scale = self._step_embeds(batch)
        loss = symmetric_infonce(img_emb, txt_emb, logit_scale)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self._val_image_emb_buf.append(img_emb.detach().float().cpu())
        self._val_text_emb_buf.append(txt_emb.detach().float().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_image_emb_buf:
            return
        img = torch.cat(self._val_image_emb_buf, dim=0)  # [N, D]
        txt = torch.cat(self._val_text_emb_buf, dim=0)   # [N, D]
        # Cheap retrieval check: each text query -> its own image among all val images
        scores = txt @ img.t()  # [N, N]
        n = scores.size(0)
        targets = torch.arange(n)
        topk = scores.topk(min(10, n), dim=1).indices  # [N, k]
        hits = (topk == targets.unsqueeze(1)).any(dim=1).float()
        self.log("val/recall@10", hits.mean(), prog_bar=True)
        self._val_image_emb_buf = []
        self._val_text_emb_buf = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.98),
        )
        if self.hparams.total_steps:
            warmup = int(self.hparams.total_steps * self.hparams.warmup_ratio)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(
                    (step + 1) / max(1, warmup),
                    0.5 * (1 + torch.cos(torch.tensor(
                        3.14159 * (step - warmup) / max(1, self.hparams.total_steps - warmup)
                    )).item())
                    if step >= warmup else (step + 1) / max(1, warmup),
                ),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return optimizer
