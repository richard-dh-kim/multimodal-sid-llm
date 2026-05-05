from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from sid_llm.data.build_cpt_corpus import (
    build_metadata_sequence,
    build_behavior_sequence,
    aggregate_user_events,
)


def test_metadata_sequence_format():
    item = {
        "title": "DeWalt 20V Drill",
        "sub_category": "Power & Hand Tools",
        "description": "Lightweight cordless.",
        "sid_0": 30, "sid_1": 156, "sid_2": 615, "sid_3": 632,
    }
    inp, tgt = build_metadata_sequence(item)
    assert inp.startswith("<seq>")
    assert "DeWalt 20V Drill" in inp
    assert "Power & Hand Tools" in inp
    assert "Lightweight cordless." in inp
    assert tgt == "<sid_30><sid_156><sid_615><sid_632><sid_eos>"


def test_metadata_sequence_truncates_long_description():
    long = "x" * 2000
    item = {"title": "T", "sub_category": "C", "description": long,
            "sid_0": 1, "sid_1": 2, "sid_2": 3, "sid_3": 4}
    inp, tgt = build_metadata_sequence(item)
    assert len(inp) < 1000  # truncated, not 2000+


def test_metadata_sequence_handles_missing_fields():
    item = {"title": "T", "sub_category": None, "description": None,
            "sid_0": 1, "sid_1": 2, "sid_2": 3, "sid_3": 4}
    inp, tgt = build_metadata_sequence(item)
    assert "<seq>" in inp
    assert tgt == "<sid_1><sid_2><sid_3><sid_4><sid_eos>"


def test_behavior_sequence_predicts_last_item():
    # user has 3 events with SIDs (1,2,3,4), (5,6,7,8), (9,10,11,12) in time order
    sid_chains = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
    ]
    inp, tgt = build_behavior_sequence(sid_chains)
    # input encodes prior items 0 and 1; target is item 2's SIDs
    assert "<sid_1>" in inp and "<sid_8>" in inp  # from prior items
    assert "<sid_9>" not in inp  # held out
    assert tgt == "<sid_9><sid_10><sid_11><sid_12><sid_eos>"


def test_behavior_sequence_returns_none_when_too_short():
    """User with fewer than 2 events can't form (history, target)."""
    assert build_behavior_sequence([(1, 2, 3, 4)]) is None
    assert build_behavior_sequence([]) is None


def test_aggregate_user_events_sorts_by_time_and_caps_length():
    rows = [
        {"user_id": "u1", "parent_asin": "B", "timestamp_ms": 200},
        {"user_id": "u1", "parent_asin": "A", "timestamp_ms": 100},
        {"user_id": "u1", "parent_asin": "C", "timestamp_ms": 300},
        {"user_id": "u2", "parent_asin": "X", "timestamp_ms": 1},
    ]
    asin_to_sid = {
        "A": (1, 2, 3, 4),
        "B": (5, 6, 7, 8),
        "C": (9, 10, 11, 12),
        "X": (99, 99, 99, 99),
    }
    user_chains = aggregate_user_events(rows, asin_to_sid, cap=10)
    assert "u1" in user_chains
    # u1's chain in time order: A, B, C
    assert user_chains["u1"] == [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
    # u2 only has 1 event — should be present (filtering happens later in build_behavior_sequence)
    assert user_chains["u2"] == [(99, 99, 99, 99)]
