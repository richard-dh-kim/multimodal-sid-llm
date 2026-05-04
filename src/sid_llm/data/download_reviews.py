"""Stream review files, filter to interactions whose parent_asin is in our items.parquet,
write Parquet of (user_id, parent_asin, rating, timestamp_ms, helpful_vote, verified_purchase).
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.request
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq

CATEGORIES: tuple[str, ...] = (
    "Tools_and_Home_Improvement",
    "Home_and_Kitchen",
    "Electronics",
)
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories"


def review_record_to_row(rec: dict) -> dict:
    return {
        "parent_asin": rec.get("parent_asin", "") or "",
        "user_id": rec.get("user_id", "") or "",
        "rating": float(rec.get("rating") or 0.0),
        "timestamp_ms": int(rec.get("timestamp") or 0),
        "helpful_vote": int(rec.get("helpful_vote") or 0),
        "verified_purchase": bool(rec.get("verified_purchase", False)),
    }


def load_item_asin_set(items_parquet: Path) -> set[str]:
    table = pq.read_table(str(items_parquet), columns=["parent_asin"])
    return set(table.column("parent_asin").to_pylist())


def stream_reviews_for_category(
    category: str,
    keep_asins: set[str],
    log_every: int = 500_000,
) -> list[dict]:
    url = f"{BASE_URL}/{category}.jsonl.gz"
    print(f"\n[{category}] Streaming reviews from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    kept: list[dict] = []
    seen = 0
    start = time.time()

    with urllib.request.urlopen(req, timeout=600) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for raw_line in gz:
                seen += 1
                if seen % log_every == 0:
                    elapsed = time.time() - start
                    rate = seen / max(elapsed, 0.001)
                    print(
                        f"  seen={seen:>10,}  kept={len(kept):>9,}  "
                        f"rate={rate:>6,.0f}/s  ({elapsed:.0f}s)"
                    )
                try:
                    rec = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                pasin = rec.get("parent_asin")
                if pasin not in keep_asins:
                    continue
                kept.append(review_record_to_row(rec))

    elapsed = time.time() - start
    print(f"[{category}] DONE: kept={len(kept):,}/{seen:,} ({elapsed:.0f}s)")
    return kept


@click.command()
@click.option(
    "--items-in", default="data/catalog/items.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out", default="data/catalog/interactions.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
def main(items_in: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading item parent_asins from {items_in} ...")
    keep_asins = load_item_asin_set(items_in)
    print(f"  {len(keep_asins):,} unique parent_asins")

    rows: list[dict] = []
    for cat in CATEGORIES:
        rows.extend(stream_reviews_for_category(cat, keep_asins))

    print(f"\n=== Total interactions: {len(rows):,} ===")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out), compression="snappy")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {size_mb:.1f} MB to {out}.")


if __name__ == "__main__":
    main()
