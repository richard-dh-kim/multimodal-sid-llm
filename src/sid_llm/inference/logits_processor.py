"""HuggingFace LogitsProcessor that constrains beam search to valid SID continuations.

The Trie was built over codebook indices [0, 1024). The model's vocabulary
includes SID tokens at IDs `sid_token_ids[i]` for codebook index `i`. At each
decode step, we:
  1. Translate the partial decode (token IDs) -> codebook index prefix
  2. Ask the Trie which codebook indices are valid next
  3. Mask the model's logits so only the corresponding SID token IDs survive
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor

from sid_llm.inference.trie import SIDTrie


_VERY_NEGATIVE = -1.0e9


class TrieConstrainedSIDProcessor(LogitsProcessor):
    """Force beam search to only emit token IDs that complete a valid SID prefix.

    Args:
        trie: SIDTrie built over codebook indices [0, codebook_size).
        sid_token_ids: list of 1024 token IDs in vocab order; sid_token_ids[i] is the
            T5 token ID for codebook index i.
        decoder_start_offset: number of leading tokens in T5's decoder that
            precede the first SID. T5 generates with a `<pad>` decoder_start_token,
            so the first generated position = step 0 = SID slot 0. Default 1
            (consume the start token before SID slot 0).
    """

    def __init__(
        self,
        trie: SIDTrie,
        sid_token_ids: list[int],
        decoder_start_offset: int = 1,
    ):
        self.trie = trie
        self.sid_token_ids = sid_token_ids
        self.token_id_to_codebook_index: dict[int, int] = {
            tid: i for i, tid in enumerate(sid_token_ids)
        }
        self.allowed_token_ids = set(sid_token_ids)
        self.decoder_start_offset = decoder_start_offset
        # Pre-build a tensor for fast vocab-wide masking.
        # For each codebook level slot, allowed token mask depends on prefix.
        self._sid_id_tensor = torch.tensor(sid_token_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """input_ids: [B*K, T] where T = decoder positions decoded so far.
        scores: [B*K, V] logits for the next token. Returns masked scores.
        """
        # Decoded SID slot for each beam = T - decoder_start_offset
        # (decoder starts with one pad token; SID slot 0 is the first generated step)
        T = input_ids.size(1)
        slot = T - self.decoder_start_offset
        if slot < 0 or slot >= 4:
            # Outside the SID generation window - don't constrain (could be the eos slot).
            return scores

        new_scores = torch.full_like(scores, _VERY_NEGATIVE)

        # Translate each beam's prefix (positions decoder_start_offset..T-1) to codebook indices.
        prefix_token_ids = input_ids[:, self.decoder_start_offset:]  # [B*K, slot]
        for beam_idx in range(input_ids.size(0)):
            prefix: tuple[int, ...] = tuple(
                self.token_id_to_codebook_index[int(t)]
                for t in prefix_token_ids[beam_idx].tolist()
                if int(t) in self.token_id_to_codebook_index
            )
            mask = self.trie.allowed_mask(prefix)  # [vocab_size_of_codebook]
            allowed_codebook_idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            if not allowed_codebook_idxs:
                # Trie reports no valid continuation (degenerate prefix). Allow all SID tokens
                # so generation doesn't stall, then the dict.get-fallback handles silent miss.
                allowed_token_ids = self.allowed_token_ids
            else:
                allowed_token_ids = {self.sid_token_ids[i] for i in allowed_codebook_idxs}

            for tid in allowed_token_ids:
                new_scores[beam_idx, tid] = scores[beam_idx, tid]

        return new_scores
