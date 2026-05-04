"""Stream Amazon Reviews 2023 metadata for our 3 categories, filter, write Parquet.

Productionized from research/_download_amazon_meta.py prototype.
Streams gzipped JSONL straight from the URL — no full download.
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import click
import pyarrow as pa
import pyarrow.parquet as pq

CATEGORIES: tuple[str, ...] = (
    "Tools_and_Home_Improvement",
    "Home_and_Kitchen",
    "Electronics",
)

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"

# Storage-room-relevant subcategories per main category (matched against categories[1]).
SUBCATEGORY_WHITELIST: dict[str, set[str]] = {
    "Tools_and_Home_Improvement": {
        "Power & Hand Tools",
        "Hardware",
        "Safety & Security",
        "Electrical",
        "Light Bulbs",
    },
    "Home_and_Kitchen": {
        "Kitchen & Dining",
        "Storage & Organization",
    },
    "Electronics": {
        "Computers & Accessories",
        "Headphones, Earbuds & Accessories",
        "Portable Audio & Video",
        "Accessories & Supplies",
        "Wearable Technology",
    },
}


def coerce_price(p: Any) -> float | None:
    """Amazon price values can be float, int, str (e.g. '$1,299.99'), or None."""
    if p is None:
        return None
    if isinstance(p, (int, float)):
        return float(p)
    if isinstance(p, str):
        s = p.strip().replace("$", "").replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def passes_quality_filter(rec: dict) -> bool:
    if (rec.get("rating_number") or 0) < 5:
        return False
    images = rec.get("images") or []
    if not isinstance(images, list):
        return False
    has_large = any(isinstance(img, dict) and img.get("large") for img in images)
    if not has_large:
        return False
    title = (rec.get("title") or "").strip()
    if len(title) < 5:
        return False
    return True


def passes_subcategory_filter(rec: dict, category: str) -> bool:
    cats = rec.get("categories") or []
    if not isinstance(cats, list) or len(cats) < 2:
        return False
    return cats[1] in SUBCATEGORY_WHITELIST.get(category, set())


def _first_large_url(images: list) -> str | None:
    for img in images:
        if isinstance(img, dict) and img.get("large"):
            return img["large"]
    return None


def _record_to_row(rec: dict, category: str) -> dict:
    cats = rec.get("categories") or []
    desc_list = rec.get("description") or []
    feat_list = rec.get("features") or []
    return {
        "main_category": category,
        "parent_asin": rec.get("parent_asin", "") or "",
        "title": (rec.get("title") or "").strip(),
        "description": (" ".join(desc_list)[:2000]) if desc_list else "",
        "features": (" | ".join(feat_list)[:2000]) if feat_list else "",
        "price": coerce_price(rec.get("price")),
        "average_rating": float(rec.get("average_rating") or 0.0),
        "rating_number": int(rec.get("rating_number") or 0),
        "store": rec.get("store") or "",
        "categories": cats if isinstance(cats, list) else [],
        "sub_category": cats[1] if isinstance(cats, list) and len(cats) > 1 else "",
        "image_url": _first_large_url(rec.get("images") or []) or "",
    }


def stream_category(category: str, max_keep: int, log_every: int = 50_000) -> list[dict]:
    """Stream and filter one category from the source URL. Stops at `max_keep`."""
    url = f"{BASE_URL}/meta_{category}.jsonl.gz"
    print(f"\n[{category}] Streaming {url}")
    print(f"  whitelist: {sorted(SUBCATEGORY_WHITELIST.get(category, set()))}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    kept: list[dict] = []
    seen = 0
    start = time.time()

    with urllib.request.urlopen(req, timeout=300) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for raw_line in gz:
                seen += 1
                if seen % log_every == 0:
                    elapsed = time.time() - start
                    rate = seen / max(elapsed, 0.001)
                    print(
                        f"  seen={seen:>9,}  kept={len(kept):>6,}  "
                        f"rate={rate:>6,.0f}/s  ({elapsed:.0f}s)"
                    )
                if len(kept) >= max_keep:
                    break

                try:
                    rec = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                if not passes_subcategory_filter(rec, category):
                    continue
                if not passes_quality_filter(rec):
                    continue

                kept.append(_record_to_row(rec, category))

    elapsed = time.time() - start
    print(f"[{category}] DONE: kept={len(kept):,}/{seen:,} ({elapsed:.0f}s)")
    return kept


@click.command()
@click.option("--max-per-category", default=50_000, type=int)
@click.option(
    "--out", default="data/catalog/items.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
def main(max_per_category: int, out: Path) -> None:
    """Download + filter all 3 categories and write to a single Parquet."""
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cat in CATEGORIES:
        rows.extend(stream_category(cat, max_per_category))

    print(f"\n=== Total: {len(rows):,} items ===")
    print(f"Writing Parquet to {out} ...")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out), compression="snappy")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Done. {size_mb:.1f} MB on disk.")


if __name__ == "__main__":
    main()
