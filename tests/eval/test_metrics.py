import numpy as np
from sid_llm.eval.metrics import recall_at_k, ndcg_at_k


def test_recall_at_k_hits():
    preds = [[1, 2, 3, 4, 5], [8, 9, 7, 6, 5]]
    targets = [3, 8]
    assert recall_at_k(preds, targets, k=1) == 0.5  # only second hits at top-1
    assert recall_at_k(preds, targets, k=3) == 1.0


def test_recall_at_k_misses():
    preds = [[10, 20, 30]]
    targets = [99]
    assert recall_at_k(preds, targets, k=3) == 0.0


def test_ndcg_at_k_perfect():
    preds = [[1, 2, 3]]
    targets = [1]
    assert ndcg_at_k(preds, targets, k=3) == 1.0


def test_ndcg_at_k_position_2():
    preds = [[2, 1, 3]]
    targets = [1]
    # 1/log2(3) / 1.0
    assert abs(ndcg_at_k(preds, targets, k=3) - (1 / np.log2(3))) < 1e-6
