"""SID-LLM Lightning module: T5 with expanded vocab for Semantic IDs.

Vocabulary additions (1028 new tokens):
  - 1024 SID tokens: <sid_0> ... <sid_1023>
  - 2 control tokens: <sid_pad>, <sid_eos>
  - 2 mode markers: <seq>, <search>

The module also owns:
  - query_projection: linear 512 -> d_model for soft-prompting from VL-CLIP query embeddings
  - soft_prompt_offsets: per-position learned offsets, [NUM_SOFT_PROMPT_TOKENS, d_model]

CPT and generative-retrieval fine-tune live in separate training scripts (M3.6, M3.7).
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5TokenizerFast


DEFAULT_MODEL = "t5-base"
QUERY_EMBED_DIM = 512   # CLIP ViT-B/32 output dim
NUM_SOFT_PROMPT_TOKENS = 4
NEW_INIT_NOISE = 0.01

# Construct the list of 1028 new tokens once at module load.
SID_TOKEN_NAMES: list[str] = (
    [f"<sid_{i}>" for i in range(1024)]
    + ["<sid_pad>", "<sid_eos>", "<seq>", "<search>"]
)


def _expand_tokenizer(tokenizer) -> int:
    """Add SID/control/mode tokens. Returns count of tokens actually added."""
    added = tokenizer.add_tokens(SID_TOKEN_NAMES, special_tokens=True)
    return added


def _init_new_rows_from_mean(weight: torch.Tensor, n_new: int, noise_std: float) -> None:
    """Overwrite the LAST n_new rows of `weight` (an embedding matrix) with
    mean(existing rows) + N(0, noise_std).
    Operates in-place; preserves dtype and device.
    """
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
    """T5-base wrapper with SID vocabulary + soft-prompt projection.

    Constructor:
      - Loads pretrained T5
      - Expands tokenizer + model embeddings
      - Initializes new rows from mean(existing) + small Gaussian noise
      - Builds query_projection and soft_prompt_offsets for search-mode inputs

    Training step / generation are added later by M3.6 (CPT) and M3.7 (gen-retrieval finetune).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        query_embed_dim: int = QUERY_EMBED_DIM,
        num_soft_prompt_tokens: int = NUM_SOFT_PROMPT_TOKENS,
        new_init_noise: float = NEW_INIT_NOISE,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load T5 + tokenizer.
        self.tokenizer = T5TokenizerFast.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)

        # Expand vocab.
        added = _expand_tokenizer(self.tokenizer)
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Re-init the appended rows (HF gives them random init by default).
        _init_new_rows_from_mean(
            self.model.shared.weight, n_new=added, noise_std=new_init_noise
        )
        # If lm_head is separate from shared (when tie_word_embeddings=False), init it too.
        if not self.model.config.tie_word_embeddings:
            _init_new_rows_from_mean(
                self.model.lm_head.weight, n_new=added, noise_std=new_init_noise
            )

        # Cache useful token ids for downstream code.
        self.sid_token_ids: list[int] = self.tokenizer.convert_tokens_to_ids(
            [f"<sid_{i}>" for i in range(1024)]
        )
        self.sid_pad_id: int = self.tokenizer.convert_tokens_to_ids("<sid_pad>")
        self.sid_eos_id: int = self.tokenizer.convert_tokens_to_ids("<sid_eos>")
        self.seq_id: int = self.tokenizer.convert_tokens_to_ids("<seq>")
        self.search_id: int = self.tokenizer.convert_tokens_to_ids("<search>")

        # Soft-prompt machinery for search-mode inputs.
        d_model = self.model.config.d_model
        self.query_projection = nn.Linear(query_embed_dim, d_model, bias=True)
        self.soft_prompt_offsets = nn.Parameter(
            torch.randn(num_soft_prompt_tokens, d_model) * new_init_noise
        )

    @property
    def num_added_tokens(self) -> int:
        return len(SID_TOKEN_NAMES)

    def soft_prompt_from_query(self, query_embed: torch.Tensor) -> torch.Tensor:
        """Build N virtual tokens for search-mode soft prompting.

        Args:
            query_embed: [B, query_embed_dim] L2-normalized VL-CLIP embedding per query.

        Returns:
            [B, N, d_model] soft prompt embeddings ready to feed into the encoder.
        """
        # [B, d_model]
        projected = self.query_projection(query_embed)
        # Broadcast + per-position offset: [B, 1, d_model] + [N, d_model] -> [B, N, d_model]
        return projected.unsqueeze(1) + self.soft_prompt_offsets.unsqueeze(0)

    def forward(self, **kwargs):
        return self.model(**kwargs)
