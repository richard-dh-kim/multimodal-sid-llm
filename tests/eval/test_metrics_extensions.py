from sid_llm.eval.metrics import hallucination_rate, silent_miss_rate


def test_hallucination_rate_zero_when_all_valid():
    valid = {(1, 2, 3, 4), (5, 6, 7, 8)}
    gen = [(1, 2, 3, 4), (5, 6, 7, 8)]
    assert hallucination_rate(gen, valid) == 0.0


def test_hallucination_rate_one_when_all_invalid():
    valid = {(1, 2, 3, 4)}
    gen = [(9, 9, 9, 9), (8, 8, 8, 8)]
    assert hallucination_rate(gen, valid) == 1.0


def test_hallucination_rate_partial():
    valid = {(1, 2, 3, 4)}
    gen = [(1, 2, 3, 4), (9, 9, 9, 9), (1, 2, 3, 4), (8, 8, 8, 8)]
    assert hallucination_rate(gen, valid) == 0.5


def test_silent_miss_rate_zero_when_all_in_dict():
    sid_to_item = {(1, 2, 3, 4): 0, (5, 6, 7, 8): 1}
    gen = [(1, 2, 3, 4), (5, 6, 7, 8)]
    assert silent_miss_rate(gen, sid_to_item) == 0.0


def test_silent_miss_rate_partial():
    sid_to_item = {(1, 2, 3, 4): 0}
    gen = [(1, 2, 3, 4), (9, 9, 9, 9)]
    assert silent_miss_rate(gen, sid_to_item) == 0.5


def test_metrics_handle_empty_inputs():
    assert hallucination_rate([], set()) == 0.0
    assert silent_miss_rate([], {}) == 0.0
