#!/usr/bin/env python3
"""
Pre-download all images from the image search cache.
Run this script on a server with international network access (e.g., AutoDL).

Usage:
    python scripts/pre_download_images.py

Images are saved to data/FVQA/downloaded_images/ as {md5_hash}.jpg
After downloading, pack the directory:
    tar czf downloaded_images.tar.gz -C data/FVQA downloaded_images/
"""

import os
import sys
import pickle
import hashlib
import time
import requests
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../data/FVQA")
TRAIN_CACHE = os.path.join(CACHE_DIR, "fvqa_train_image_search_results_cache.pkl")
TEST_CACHE = os.path.join(CACHE_DIR, "fvqa_test_image_search_results_cache.pkl")
OUTPUT_DIR = os.path.join(CACHE_DIR, "downloaded_images")


def url_to_path(url: str) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(OUTPUT_DIR, f"{url_hash}.jpg")


def download_one(url: str) -> tuple[str, bool]:
    """Download a single image. Returns (url, success)."""
    out_path = url_to_path(url)
    if os.path.exists(out_path):
        return url, True
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(out_path, "JPEG", quality=95)
        return url, True
    except Exception as e:
        return url, False


def main():
    # Load caches
    all_urls = set()
    for cache_path in [TRAIN_CACHE, TEST_CACHE]:
        if not os.path.exists(cache_path):
            print(f"Cache not found: {cache_path}")
            continue
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        for entry in cache.values():
            for url in entry.get("tool_returned_images_urls", []):
                if isinstance(url, str) and url.startswith("http"):
                    all_urls.add(url)
        print(f"Loaded {cache_path}: {len(cache)} entries")

    print(f"\nTotal unique image URLs: {len(all_urls)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Filter out already downloaded
    to_download = [u for u in all_urls if not os.path.exists(url_to_path(u))]
    already = len(all_urls) - len(to_download)
    print(f"Already downloaded: {already}")
    print(f"To download: {len(to_download)}")

    if not to_download:
        print("All images already downloaded!")
        return

    # Download with thread pool
    success = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(download_one, url): url for url in to_download}
        for i, future in enumerate(as_completed(futures)):
            url, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (success + failed) / elapsed
                print(f"  Progress: {success + failed}/{len(to_download)} ({rate:.1f}/s), success={success}, failed={failed}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Success: {success}, Failed: {failed}")
    print(f"Images saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"\nTo pack for transfer:")
    print(f"  tar czf downloaded_images.tar.gz -C {os.path.abspath(CACHE_DIR)} downloaded_images/")


if __name__ == "__main__":
    main()
