"""Parallel image downloader with retries.

Reads `items.parquet`, downloads the `image_url` for each row to
`data/images/{shard}/{item_id}.jpg`. Uses item_id = row index in the Parquet.

Concurrency: ~32 in-flight requests. Retries transient failures (5xx, timeouts).
Logs permanent failures to a Parquet file for later inspection. Skips already-
downloaded items so the script is idempotent and resumable.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Iterable

import aiohttp
import click
import pyarrow as pa
import pyarrow.parquet as pq

CONCURRENCY = 32
TIMEOUT_SECONDS = 30
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 1.5  # seconds, exponential


def derive_item_id_from_row_index(row_idx: int) -> int:
    return row_idx


def image_path_for_item(images_dir: Path, item_id: int) -> Path:
    """Shard files into <=1000 directories. ids < 1000 go in shard f"{id:03d}";
    ids >= 1000 go in shard f"{id // 1000:03d}". Caps at id < 10^6.
    Examples: id=0 → "000/0.jpg", id=10 → "010/10.jpg", id=12345 → "012/12345.jpg".
    """
    if item_id < 1000:
        shard = f"{item_id:03d}"
    else:
        shard = f"{item_id // 1000:03d}"
    return images_dir / shard / f"{item_id}.jpg"


def is_already_downloaded(images_dir: Path, item_id: int) -> bool:
    p = image_path_for_item(images_dir, item_id)
    return p.exists() and p.stat().st_size > 0


async def _download_one(
    session: aiohttp.ClientSession,
    item_id: int,
    url: str,
    images_dir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[int, bool, str]:
    """Returns (item_id, success, message)."""
    if is_already_downloaded(images_dir, item_id):
        return (item_id, True, "skipped")

    out_path = image_path_for_item(images_dir, item_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = ""
    for attempt in range(RETRY_ATTEMPTS):
        sleep_after = 0.0
        try:
            async with semaphore:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as resp:
                    if resp.status >= 500:
                        last_error = f"HTTP {resp.status}"
                        sleep_after = RETRY_BACKOFF ** attempt
                    elif resp.status >= 400:
                        return (item_id, False, f"HTTP {resp.status}")
                    else:
                        body = await resp.read()
                        if len(body) < 100:
                            return (item_id, False, f"too small ({len(body)} bytes)")
                        out_path.write_bytes(body)
                        return (item_id, True, "ok")
        except asyncio.TimeoutError:
            last_error = "timeout"
            sleep_after = RETRY_BACKOFF ** attempt
        except aiohttp.ClientError as e:
            last_error = f"client_error: {type(e).__name__}"
            sleep_after = RETRY_BACKOFF ** attempt
        if sleep_after:
            await asyncio.sleep(sleep_after)

    return (item_id, False, last_error)


async def _run_downloads(
    rows: Iterable[tuple[int, str]], images_dir: Path
) -> list[tuple[int, bool, str]]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    headers = {"User-Agent": "Mozilla/5.0 (research)"}
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = [
            _download_one(session, item_id, url, images_dir, semaphore)
            for item_id, url in rows
        ]
        results: list[tuple[int, bool, str]] = []
        for fut in asyncio.as_completed(tasks):
            r = await fut
            results.append(r)
            if len(results) % 1000 == 0:
                ok = sum(1 for _, s, _ in results if s)
                print(f"  progress: {len(results):>6,} done  ok={ok:,}  fails={len(results)-ok:,}")
        return results


@click.command()
@click.option(
    "--items-in", default="data/catalog/items.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--images-dir", default="data/images",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--failures-log", default="data/catalog/image_failures.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
def main(items_in: Path, images_dir: Path, failures_log: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    failures_log.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading items from {items_in} ...")
    table = pq.read_table(str(items_in), columns=["image_url"])
    urls = table.column("image_url").to_pylist()
    rows = [(i, u) for i, u in enumerate(urls) if u]
    print(f"  {len(rows):,} items with URLs")

    start = time.time()
    results = asyncio.run(_run_downloads(rows, images_dir))
    elapsed = time.time() - start

    ok = sum(1 for _, s, _ in results if s)
    fail = len(results) - ok
    print(f"\nDone in {elapsed:.0f}s. ok={ok:,}  failed={fail:,}")

    fails = [
        {"item_id": i, "reason": msg}
        for i, success, msg in results
        if not success
    ]
    if fails:
        ftable = pa.Table.from_pylist(fails)
        pq.write_table(ftable, str(failures_log), compression="snappy")
        print(f"  failures logged to {failures_log}")


if __name__ == "__main__":
    main()
