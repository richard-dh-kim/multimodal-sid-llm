from sid_llm.data.download_reviews import (
    review_record_to_row,
    load_item_asin_set,
)


def test_review_record_to_row_extracts_required_fields():
    rec = {
        "rating": 4.0,
        "title": "Great drill",
        "text": "Used it on a deck.",
        "asin": "B07X1234",
        "parent_asin": "B07X1234",
        "user_id": "U1",
        "timestamp": 1640000000000,
        "helpful_vote": 2,
        "verified_purchase": True,
        "images": [],
    }
    row = review_record_to_row(rec)
    assert row["parent_asin"] == "B07X1234"
    assert row["user_id"] == "U1"
    assert row["rating"] == 4.0
    assert row["timestamp_ms"] == 1640000000000
    assert row["helpful_vote"] == 2
    assert row["verified_purchase"] is True


def test_review_record_to_row_handles_missing_fields():
    rec = {"parent_asin": "B07X1234", "user_id": "U1"}
    row = review_record_to_row(rec)
    assert row["rating"] == 0.0
    assert row["timestamp_ms"] == 0


def test_load_item_asin_set_returns_set_of_asins(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    items = pa.Table.from_pylist([
        {"parent_asin": "A", "title": "x"},
        {"parent_asin": "B", "title": "y"},
        {"parent_asin": "C", "title": "z"},
    ])
    p = tmp_path / "items.parquet"
    pq.write_table(items, str(p))

    asins = load_item_asin_set(p)
    assert asins == {"A", "B", "C"}
