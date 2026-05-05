"""Standard retrieval metrics."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def recall_at_k(preds: Sequence[Sequence[int]], targets: Sequence[int], k: int) -> float:
    if not preds:
        return 0.0
    hits = 0
    for ranked, target in zip(preds, targets):
        if target in ranked[:k]:
            hits += 1
    return hits / len(preds)


def ndcg_at_k(preds: Sequence[Sequence[int]], targets: Sequence[int], k: int) -> float:
    if not preds:
        return 0.0
    total = 0.0
    for ranked, target in zip(preds, targets):
        for pos, item in enumerate(ranked[:k]):
            if item == target:
                total += 1.0 / np.log2(pos + 2)
                break
    return total / len(preds)


def hallucination_rate(
    generated_sids: list[tuple[int, ...]],
    valid_sids: set[tuple[int, ...]],
) -> float:
    """Fraction of generated SIDs that are not in `valid_sids`."""
    if not generated_sids:
        return 0.0
    miss = sum(1 for s in generated_sids if s not in valid_sids)
    return miss / len(generated_sids)


def silent_miss_rate(
    generated_sids: list[tuple[int, ...]],
    sid_to_item: dict,
) -> float:
    """Fraction of generated SIDs that don't map to an item via the lookup dict."""
    if not generated_sids:
        return 0.0
    miss = sum(1 for s in generated_sids if s not in sid_to_item)
    return miss / len(generated_sids)
