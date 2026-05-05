"""Trie for constrained beam-search decoding.

Built from a list of valid SID tuples (4-token sequences over a fixed vocab).
At each beam-search step, given the partial decoded prefix, the trie tells
the decoder which next tokens can possibly complete a valid SID. Tokens
outside the allowed set get their logits forced to -inf (or a very negative
value) before the softmax, guaranteeing that beam search never emits a tuple
that isn't in the catalog.
"""
from __future__ import annotations

import torch


# A finite very-negative value plays better with mixed precision than -inf.
_VERY_NEGATIVE = -1.0e9


class _Node:
    __slots__ = ("children", "is_terminal")

    def __init__(self):
        self.children: dict[int, _Node] = {}
        self.is_terminal: bool = False


class SIDTrie:
    """Trie of valid SID tuples for constrained decoding.

    Construction: O(N * L) where N is num catalog items, L is tokens per SID.
    Memory: each node holds a dict of children; ~200 MB for 150k SIDs is realistic.
    Lookup (mask_logits): O(L * fanout_at_node).
    """

    def __init__(self, sids: list[tuple[int, ...]], vocab_size: int):
        self.vocab_size = vocab_size
        self.root = _Node()
        for sid in sids:
            self._insert(sid)

    def _insert(self, sid: tuple[int, ...]) -> None:
        node = self.root
        for tok in sid:
            if tok not in node.children:
                node.children[tok] = _Node()
            node = node.children[tok]
        node.is_terminal = True

    def _walk(self, prefix: tuple[int, ...]) -> _Node | None:
        """Walk to the node at `prefix`, or return None if no such path exists."""
        node = self.root
        for tok in prefix:
            child = node.children.get(tok)
            if child is None:
                return None
            node = child
        return node

    def is_valid(self, sid: tuple[int, ...]) -> bool:
        """True iff `sid` was inserted (matches a leaf marked terminal)."""
        node = self._walk(sid)
        return node is not None and node.is_terminal

    def allowed_mask(self, prefix: tuple[int, ...]) -> torch.Tensor:
        """Boolean mask over the vocabulary: True at positions that are valid
        next tokens given `prefix`. If the prefix is unreachable, all-False.
        """
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        node = self._walk(prefix)
        if node is None:
            return mask
        for tok in node.children:
            if 0 <= tok < self.vocab_size:
                mask[tok] = True
        return mask

    def apply_mask(
        self,
        prefix: tuple[int, ...],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Set disallowed positions in `logits` to a very negative value.

        Accepts logits of shape `[V]` (single beam) or `[B, V]` (batched beams
        sharing the same prefix). Returns a new tensor; does not mutate input.
        """
        mask = self.allowed_mask(prefix).to(logits.device)
        if logits.dim() == 1:
            return torch.where(mask, logits, torch.full_like(logits, _VERY_NEGATIVE))
        # logits: [B, V]; broadcast mask along batch
        mask_b = mask.unsqueeze(0).expand_as(logits)
        return torch.where(mask_b, logits, torch.full_like(logits, _VERY_NEGATIVE))
