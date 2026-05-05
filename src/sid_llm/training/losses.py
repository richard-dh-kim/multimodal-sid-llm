"""Loss functions for VL-CLIP fine-tuning and beyond."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_infonce(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE loss for image-text contrastive training (CLIP-style).

    Args:
        image_embeds: [N, D] L2-normalized image embeddings.
        text_embeds:  [N, D] L2-normalized text embeddings.
        logit_scale:  scalar log temperature; final scale = exp(logit_scale).

    Returns:
        Scalar loss = 0.5 * (CE(img->txt) + CE(txt->img)).
    """
    scale = logit_scale.exp()
    logits_per_image = scale * image_embeds @ text_embeds.t()  # [N, N]
    logits_per_text = logits_per_image.t()
    n = logits_per_image.size(0)
    targets = torch.arange(n, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i + loss_t)
