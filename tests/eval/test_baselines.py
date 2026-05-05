import numpy as np
from sid_llm.eval.baselines import mips_topk


def test_mips_topk_returns_argmax():
    catalog = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    queries = np.array([
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 0.9],
    ], dtype=np.float32)
    item_ids = [10, 20, 30]
    preds = mips_topk(queries, catalog, item_ids, k=2)
    assert preds[0][0] == 10  # closest to first query
    assert preds[1][0] == 30
