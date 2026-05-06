"""T5 with expanded vocab (1024 SIDs + control + mode markers) and soft-prompt projection."""
from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5TokenizerFast


DEFAULT_MODEL = "t5-base"
QUERY_EMBED_DIM = 512   # CLIP ViT-B/32 output dim
NUM_SOFT_PROMPT_TOKENS = 4
NEW_INIT_NOISE = 0.01

SID_TOKEN_NAMES: list[str] = (
    [f"<sid_{i}>" for i in range(1024)]
    + ["<sid_pad>", "<sid_eos>", "<seq>", "<search>"]
)


def _expand_tokenizer(tokenizer) -> int:
    return tokenizer.add_tokens(SID_TOKEN_NAMES, special_tokens=True)


def _init_new_rows_from_mean(weight: torch.Tensor, n_new: int, noise_std: float) -> None:
    """Overwrite the last n_new rows of `weight` with mean(existing) + N(0, noise_std), in place."""
    if n_new <= 0:
        return
    n_total = weight.size(0)
    n_existing = n_total - n_new
    with torch.no_grad():
        mean = weight[:n_existing].mean(dim=0, keepdim=True)
        noise = torch.randn(
            n_new, weight.size(1), device=weight.device, dtype=weight.dtype
        ) * noise_std
        weight[n_existing:n_total] = mean + noise


class SIDLLMLightning(L.LightningModule):
    """T5 with expanded SID vocabulary and a query_projection / soft_prompt_offsets pair."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        query_embed_dim: int = QUERY_EMBED_DIM,
        num_soft_prompt_tokens: int = NUM_SOFT_PROMPT_TOKENS,
        new_init_noise: float = NEW_INIT_NOISE,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = T5TokenizerFast.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)

        added = _expand_tokenizer(self.tokenizer)
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        _init_new_rows_from_mean(
            self.model.shared.weight, n_new=added, noise_std=new_init_noise
        )
        if not self.model.config.tie_word_embeddings:
            _init_new_rows_from_mean(
                self.model.lm_head.weight, n_new=added, noise_std=new_init_noise
            )

        self.sid_token_ids: list[int] = self.tokenizer.convert_tokens_to_ids(
            [f"<sid_{i}>" for i in range(1024)]
        )
        self.sid_pad_id: int = self.tokenizer.convert_tokens_to_ids("<sid_pad>")
        self.sid_eos_id: int = self.tokenizer.convert_tokens_to_ids("<sid_eos>")
        self.seq_id: int = self.tokenizer.convert_tokens_to_ids("<seq>")
        self.search_id: int = self.tokenizer.convert_tokens_to_ids("<search>")

        d_model = self.model.config.d_model
        self.query_projection = nn.Linear(query_embed_dim, d_model, bias=True)
        self.soft_prompt_offsets = nn.Parameter(
            torch.randn(num_soft_prompt_tokens, d_model) * new_init_noise
        )

    @property
    def num_added_tokens(self) -> int:
        return len(SID_TOKEN_NAMES)

    def soft_prompt_from_query(self, query_embed: torch.Tensor) -> torch.Tensor:
        """[B, query_embed_dim] -> [B, N, d_model]."""
        projected = self.query_projection(query_embed)
        return projected.unsqueeze(1) + self.soft_prompt_offsets.unsqueeze(0)

    def forward(self, **kwargs):
        return self.model(**kwargs)
