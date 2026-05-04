"""Vanilla CLIP embedder (ViT-B/32).

Computes per-item fused multimodal embeddings via L2-averaging of the image
and text embeddings (parameter-free fusion, since contrastive InfoNCE pulls
them into the same shared space). Used for the B1 baseline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class ClipEmbedder:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.embed_dim = self.model.config.projection_dim  # 512 for ViT-B/32

    @torch.no_grad()
    def embed_image_text_batch(
        self, images: list[Image.Image], texts: list[str]
    ) -> torch.Tensor:
        inputs = self.processor(
            images=images, text=texts, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        ).to(self.device)
        outputs = self.model(**inputs)
        v_img = outputs.image_embeds  # already L2-normed
        v_txt = outputs.text_embeds   # already L2-normed
        v = F.normalize((v_img + v_txt) / 2.0, dim=-1)
        return v.cpu()

    @torch.no_grad()
    def embed_image_text(self, image: Image.Image, text: str) -> torch.Tensor:
        return self.embed_image_text_batch([image], [text])[0]
