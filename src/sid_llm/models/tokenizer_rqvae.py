"""RQ-VAE tokenizer for Semantic ID generation.

Wraps `vector_quantize_pytorch.ResidualVQ` to quantize a per-item embedding
(typically 512-D from CLIP) into a 4-token tuple drawn from a 1024-entry
codebook per level. The output indices ARE the item's Semantic ID.

Trained via reconstruction MSE + commitment loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ


class RQVAETokenizer(nn.Module):
    """Residual quantization VAE-style tokenizer.

    Args:
        dim: input embedding dimension. Default 512 for CLIP ViT-B/32.
        num_quantizers: how many residual codebooks to stack. Default 4 (matches TIGER/PLUM).
        codebook_size: entries per codebook. Default 1024.
        commitment_weight: weight on the commitment loss term (β in VQ-VAE). Default 0.25.
        kmeans_init: initialize codebooks via k-means on the first batch (improves stability).
        decay: EMA decay for codebook updates.
    """

    def __init__(
        self,
        dim: int = 512,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        commitment_weight: float = 0.25,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        decay: float = 0.99,
    ):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.vq = ResidualVQ(
            dim=dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a batch of embeddings.

        Args:
            x: [B, D] input embeddings.

        Returns:
            quantized: [B, D] reconstructed (quantized) embeddings.
            indices:   [B, num_quantizers] integer codebook indices (the SID tuple).
            commit_loss: scalar commitment loss summed across quantizer levels.
        """
        # ResidualVQ expects [B, *, D]; add a length-1 sequence axis.
        x_seq = x.unsqueeze(1)
        quantized, indices, commit_losses = self.vq(x_seq)
        quantized = quantized.squeeze(1)  # [B, D]
        indices = indices.squeeze(1)      # [B, num_quantizers]
        # commit_losses can be [B, Q] or [Q]; reduce to scalar.
        if commit_losses.dim() > 0:
            commit_loss = commit_losses.sum()
        else:
            commit_loss = commit_losses
        return quantized, indices, commit_loss
