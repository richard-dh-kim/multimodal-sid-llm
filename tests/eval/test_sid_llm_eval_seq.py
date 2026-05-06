"""CPU-only tests for sid_llm_eval_seq's query-construction helper.

These tests do NOT load a model; they exercise `_build_seq_queries` and
`_history_to_query_text` against synthetic interaction frames so the eval
format (and held-out target semantics) can be validated cheaply.
"""
from __future__ import annotations

import pandas as pd
import pytest

from sid_llm.data.build_cpt_corpus import build_behavior_sequence
from sid_llm.eval.sid_llm_eval_seq import (
    _build_seq_queries,
    _history_to_query_text,
)


def _make_interactions(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    """rows: list of (user_id, parent_asin, timestamp_ms)."""
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "timestamp_ms"])


def _trivial_catalog(asins: list[str]) -> tuple[
    dict[str, int], dict[int, tuple[int, int, int, int]]
]:
    """Assign item_ids 0..N-1 and a unique 4-tuple SID per ASIN."""
    asin_to_iid = {a: i for i, a in enumerate(asins)}
    iid_to_sid = {
        i: (i, i + 100, i + 200, i + 300) for i in range(len(asins))
    }
    return asin_to_iid, iid_to_sid


def test_query_construction_holds_out_last_item():
    asins = ["A", "B", "C", "D"]
    asin_to_iid, iid_to_sid = _trivial_catalog(asins)
    df = _make_interactions([
        ("u1", "A", 100),
        ("u1", "B", 200),
        ("u1", "C", 300),
        ("u1", "D", 400),
    ])

    queries, targets = _build_seq_queries(
        df, asin_to_iid, iid_to_sid,
        max_queries=10, min_history=2, max_history_len=50, seed=0,
    )

    assert len(queries) == 1
    assert len(targets) == 1
    # Target must be item D (last interaction).
    assert targets[0] == asin_to_iid["D"]
    # History must be exactly A, B, C — the first three SIDs in order.
    expected = "<seq> "
    for asin in ("A", "B", "C"):
        for c in iid_to_sid[asin_to_iid[asin]]:
            expected += f"<sid_{c}>"
    assert queries[0] == expected
    # Target item D's SIDs should NOT appear in the history string.
    d_sid = iid_to_sid[asin_to_iid["D"]]
    assert f"<sid_{d_sid[0]}>" + f"<sid_{d_sid[1]}>" not in queries[0]


def test_query_clips_to_max_history_len():
    n = 100
    asins = [f"A{i:03d}" for i in range(n)]
    asin_to_iid, iid_to_sid = _trivial_catalog(asins)
    df = _make_interactions([("u1", a, 100 + i) for i, a in enumerate(asins)])

    max_hist = 10
    queries, targets = _build_seq_queries(
        df, asin_to_iid, iid_to_sid,
        max_queries=10, min_history=2, max_history_len=max_hist, seed=0,
    )

    assert len(queries) == 1
    # Target = last item; history before clipping = first 99; after clipping = last 10 of those.
    assert targets[0] == asin_to_iid[asins[-1]]
    q = queries[0]
    # The query must contain SIDs for items at positions [n-1-max_hist .. n-2]
    # (i.e., the 10 most-recent items in the pre-target history).
    expected_history_iids = [asin_to_iid[asins[i]] for i in range(n - 1 - max_hist, n - 1)]
    expected_inner = "".join(
        "".join(f"<sid_{c}>" for c in iid_to_sid[iid])
        for iid in expected_history_iids
    )
    assert q == f"<seq> {expected_inner}"
    # SIDs from items earlier than the clip window must be absent.
    too_old_iid = asin_to_iid[asins[n - 2 - max_hist]]
    too_old_first_sid = iid_to_sid[too_old_iid][0]
    # The first SID component is unique to that item under our trivial catalog,
    # so its absence proves the clip happened.
    assert f"<sid_{too_old_first_sid}>" not in q


def test_skips_users_below_min_history():
    asins = ["A", "B", "C", "D", "E"]
    asin_to_iid, iid_to_sid = _trivial_catalog(asins)
    df = _make_interactions([
        # u_short: 2 events total -> history_len=1, below min_history=2.
        ("u_short", "A", 100),
        ("u_short", "B", 200),
        # u_ok: 3 events total -> history_len=2, exactly meets min_history=2.
        ("u_ok", "A", 100),
        ("u_ok", "B", 200),
        ("u_ok", "C", 300),
        # u_big: 4 events total -> well above threshold.
        ("u_big", "A", 100),
        ("u_big", "B", 200),
        ("u_big", "C", 300),
        ("u_big", "D", 400),
    ])

    queries, targets = _build_seq_queries(
        df, asin_to_iid, iid_to_sid,
        max_queries=100, min_history=2, max_history_len=50, seed=0,
    )

    # u_short must be dropped; u_ok and u_big must both be kept.
    assert len(queries) == 2
    assert len(targets) == 2


def test_query_format_matches_cpt_corpus():
    """The eval CLI's query string must match build_cpt_corpus.build_behavior_sequence
    byte-for-byte on the input side."""
    asins = ["A", "B", "C", "D"]
    asin_to_iid, iid_to_sid = _trivial_catalog(asins)
    df = _make_interactions([
        ("u1", "A", 100),
        ("u1", "B", 200),
        ("u1", "C", 300),
        ("u1", "D", 400),
    ])

    queries, targets = _build_seq_queries(
        df, asin_to_iid, iid_to_sid,
        max_queries=10, min_history=2, max_history_len=50, seed=0,
    )
    assert len(queries) == 1

    # Build the equivalent SID chain (full 4 items, last held out by build_behavior_sequence)
    # and compare the input string to the eval query.
    full_chain = [iid_to_sid[asin_to_iid[a]] for a in asins]
    cpt_input, _cpt_target = build_behavior_sequence(full_chain)
    assert queries[0] == cpt_input

    # Spot-check the structural format: starts with "<seq> ", no inner spaces, exact tuple count.
    assert queries[0].startswith("<seq> ")
    body = queries[0][len("<seq> "):]
    assert " " not in body
    # 3 history items * 4 SID tokens = 12 tokens.
    assert body.count("<sid_") == 12


def test_history_to_query_text_no_inner_spaces():
    history = [(1, 2, 3, 4), (5, 6, 7, 8)]
    out = _history_to_query_text(history)
    assert out == "<seq> <sid_1><sid_2><sid_3><sid_4><sid_5><sid_6><sid_7><sid_8>"


def test_max_queries_caps_user_count():
    asins = ["A", "B", "C"]
    asin_to_iid, iid_to_sid = _trivial_catalog(asins)
    rows = []
    for u in range(20):
        rows.extend([
            (f"u{u}", "A", 100),
            (f"u{u}", "B", 200),
            (f"u{u}", "C", 300),
        ])
    df = _make_interactions(rows)

    queries, _ = _build_seq_queries(
        df, asin_to_iid, iid_to_sid,
        max_queries=5, min_history=2, max_history_len=50, seed=42,
    )
    assert len(queries) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
