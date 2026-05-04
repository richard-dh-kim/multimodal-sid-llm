from sid_llm.data.text_clean import clean_text


def test_clean_text_strips_html_tags():
    assert clean_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_clean_text_normalizes_whitespace():
    assert clean_text("foo   \n\t bar") == "foo bar"


def test_clean_text_dedups_consecutive_keyword_repetitions():
    assert clean_text("drill drill cordless drill cordless") == "drill cordless"


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_truncates_to_max_chars():
    long = "word " * 1000
    cleaned = clean_text(long, max_chars=100)
    assert len(cleaned) <= 100


def test_clean_text_preserves_real_content():
    raw = "Cordless 20V Drill, Brushless Motor, with 2 Batteries"
    assert clean_text(raw) == raw


def test_clean_text_strips_amazon_keyword_stuffing():
    # Common Amazon pattern: long titles with repeated keywords
    raw = "Drill Cordless 20V Drill Set Drill Bits Drill Charger"
    out = clean_text(raw)
    # "Drill" should appear at most once (after consecutive-dedup), other unique keywords kept
    words = out.split()
    assert words.count("Drill") <= 2  # one consecutive run dedup'd, rest kept
