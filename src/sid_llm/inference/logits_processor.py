"""LogitsProcessor that constrains beam search to valid SID continuations from a Trie."""
from __future__ import annotations

import torch
from transformers import LogitsProcessor

from sid_llm.inference.trie import SIDTrie


_VERY_NEGATIVE = -1.0e9


class TrieConstrainedSIDProcessor(LogitsProcessor):
    """Force beam search to only emit token IDs that complete a valid SID prefix.

    `decoder_start_offset` is the number of leading decoder tokens before the
    first SID slot (T5 emits one pad token first, so default 1).
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
        self._sid_id_tensor = torch.tensor(sid_token_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        T = input_ids.size(1)
        slot = T - self.decoder_start_offset
        if slot < 0 or slot >= 4:
            return scores

        new_scores = torch.full_like(scores, _VERY_NEGATIVE)

        prefix_token_ids = input_ids[:, self.decoder_start_offset:]
        for beam_idx in range(input_ids.size(0)):
            prefix: tuple[int, ...] = tuple(
                self.token_id_to_codebook_index[int(t)]
                for t in prefix_token_ids[beam_idx].tolist()
                if int(t) in self.token_id_to_codebook_index
            )
            mask = self.trie.allowed_mask(prefix)
            allowed_codebook_idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            if not allowed_codebook_idxs:
                # Degenerate prefix: allow any SID so generation continues; the
                # sid_to_item lookup will surface the resulting silent miss.
                allowed_token_ids = self.allowed_token_ids
            else:
                allowed_token_ids = {self.sid_token_ids[i] for i in allowed_codebook_idxs}

            for tid in allowed_token_ids:
                new_scores[beam_idx, tid] = scores[beam_idx, tid]

        return new_scores
