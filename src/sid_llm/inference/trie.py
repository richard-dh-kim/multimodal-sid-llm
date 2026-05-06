"""Trie of valid SID tuples for constrained beam-search decoding."""
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
    """Trie of valid SID tuples; supports prefix membership and per-prefix vocab masks."""

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
        node = self.root
        for tok in prefix:
            child = node.children.get(tok)
            if child is None:
                return None
            node = child
        return node

    def is_valid(self, sid: tuple[int, ...]) -> bool:
        node = self._walk(sid)
        return node is not None and node.is_terminal

    def allowed_mask(self, prefix: tuple[int, ...]) -> torch.Tensor:
        """Boolean [vocab_size] mask of valid next tokens after `prefix`."""
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
        """Return a copy of `logits` with disallowed positions set to a very negative value.

        Accepts shape `[V]` or `[B, V]` (batched beams sharing the prefix).
        """
        mask = self.allowed_mask(prefix).to(logits.device)
        if logits.dim() == 1:
            return torch.where(mask, logits, torch.full_like(logits, _VERY_NEGATIVE))
        mask_b = mask.unsqueeze(0).expand_as(logits)
        return torch.where(mask_b, logits, torch.full_like(logits, _VERY_NEGATIVE))
