from sid_llm.data.download_metadata import (
    coerce_price,
    passes_quality_filter,
    passes_subcategory_filter,
    SUBCATEGORY_WHITELIST,
)


def test_coerce_price_handles_string_with_dollar_sign():
    assert coerce_price("$1,299.99") == 1299.99


def test_coerce_price_returns_none_for_missing():
    assert coerce_price(None) is None
    assert coerce_price("") is None
    assert coerce_price("not a number") is None


def test_coerce_price_passes_through_numeric():
    assert coerce_price(42) == 42.0
    assert coerce_price(3.14) == 3.14


def test_passes_quality_filter_rejects_low_review_count():
    rec = {"rating_number": 4, "title": "Some Product", "images": [{"large": "u"}]}
    assert passes_quality_filter(rec) is False


def test_passes_quality_filter_rejects_no_image():
    rec = {"rating_number": 100, "title": "Some Product", "images": []}
    assert passes_quality_filter(rec) is False


def test_passes_quality_filter_rejects_no_large_url():
    rec = {
        "rating_number": 100,
        "title": "Some Product",
        "images": [{"thumb": "url"}],  # missing 'large'
    }
    assert passes_quality_filter(rec) is False


def test_passes_quality_filter_rejects_short_title():
    rec = {"rating_number": 100, "title": "AB", "images": [{"large": "u"}]}
    assert passes_quality_filter(rec) is False


def test_passes_quality_filter_accepts_valid():
    rec = {
        "rating_number": 5,
        "title": "Cordless Drill",
        "images": [{"large": "https://m.media-amazon.com/x.jpg"}],
    }
    assert passes_quality_filter(rec) is True


def test_passes_subcategory_filter_uses_categories_index_1():
    rec = {"categories": ["Tools & Home Improvement", "Power & Hand Tools", "Drills"]}
    assert passes_subcategory_filter(rec, "Tools_and_Home_Improvement") is True


def test_passes_subcategory_filter_rejects_off_whitelist():
    rec = {"categories": ["Tools & Home Improvement", "Lighting & Ceiling Fans"]}
    assert passes_subcategory_filter(rec, "Tools_and_Home_Improvement") is False


def test_passes_subcategory_filter_rejects_missing_subcategory():
    rec = {"categories": ["Tools & Home Improvement"]}
    assert passes_subcategory_filter(rec, "Tools_and_Home_Improvement") is False


def test_whitelist_contains_storage_relevant_subcategories():
    assert "Power & Hand Tools" in SUBCATEGORY_WHITELIST["Tools_and_Home_Improvement"]
    assert "Kitchen & Dining" in SUBCATEGORY_WHITELIST["Home_and_Kitchen"]
    assert "Computers & Accessories" in SUBCATEGORY_WHITELIST["Electronics"]
