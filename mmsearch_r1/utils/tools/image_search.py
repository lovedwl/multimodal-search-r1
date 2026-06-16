import os
import pickle
import hashlib
from PIL import Image


# Cache config
_TRAIN_CACHE_PATH = os.environ.get(
    "IMAGE_SEARCH_CACHE_TRAIN",
    os.path.join(os.path.dirname(__file__), "../../../data/FVQA/fvqa_train_image_search_results_cache.pkl"),
)
_TEST_CACHE_PATH = os.environ.get(
    "IMAGE_SEARCH_CACHE_TEST",
    os.path.join(os.path.dirname(__file__), "../../../data/FVQA/fvqa_test_image_search_results_cache.pkl"),
)

# Downloaded image disk cache directory
_IMAGE_DISK_CACHE_DIR = os.environ.get(
    "IMAGE_DISK_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "../../../data/FVQA/downloaded_images"),
)

# Lazy-loaded search result cache (data_id -> {titles, urls})
_search_cache = None


def _load_search_cache() -> dict:
    """Load and merge train + test image search result caches."""
    global _search_cache
    if _search_cache is not None:
        return _search_cache

    _search_cache = {}
    for path in [_TRAIN_CACHE_PATH, _TEST_CACHE_PATH]:
        path = os.path.abspath(path)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                _search_cache.update(data)
                print(f"[Image Search] Loaded search cache from {path}: {len(data)} entries")
            except Exception as e:
                print(f"[Image Search] Failed to load search cache from {path}: {e}")
        else:
            print(f"[Image Search] Search cache not found: {path}")

    print(f"[Image Search] Total search cache entries: {len(_search_cache)}")
    return _search_cache


def _url_to_cache_path(url: str) -> str:
    """Convert a URL to a local disk cache file path using MD5 hash."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(_IMAGE_DISK_CACHE_DIR, f"{url_hash}.jpg")


def _load_cached_image(url: str) -> Image.Image | None:
    """Load image from disk cache. Returns None if not cached."""
    cache_path = _url_to_cache_path(url)
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            return None
    return None


def call_image_search(image_url: str, data_id: str = None):
    """
    Image search tool: look up cached SerpAPI results by data_id,
    download thumbnails (with disk caching), and return in standard format.

    Args:
        image_url: The original image URL.
        data_id: The data_id to look up in the cache.

    Returns:
        tool_returned_str (str): Formatted search results with image tokens and titles.
        tool_returned_images (list[PIL.Image]): Downloaded thumbnail images.
        tool_stat (dict): Tool status.
    """
    print(f"[Image Search] data_id={data_id}, image_url={image_url}")

    cache = _load_search_cache()

    # Look up cache by data_id
    cache_entry = None
    if data_id and data_id in cache:
        cache_entry = cache[data_id]

    if cache_entry is None:
        tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capabilities."
        return tool_returned_str, [], {"success": False, "num_images": 0}

    titles = cache_entry.get("tool_returned_web_title_list", [])
    image_urls = cache_entry.get("tool_returned_images_urls", [])

    # Download images (with disk cache)
    tool_returned_images = []
    tool_returned_str = "[Image Search Results] The result of the image search consists of web page information related to the image from the user's original question. Each result includes the main image from the web page and its title, ranked in descending order of search relevance, as demonstrated below:\n"

    for i, (title, url) in enumerate(zip(titles, image_urls)):
        img = None
        if isinstance(url, str) and url.startswith("http"):
            img = _load_cached_image(url)
        if img is not None:
            tool_returned_images.append(img)
            tool_returned_str += f"{i+1}. image: <|vision_start|><|image_pad|><|vision_end|>\ntitle: {title}\n"
        else:
            tool_returned_str += f"{i+1}. title: {title}\n"

    if not tool_returned_images:
        tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capabilities."
        return tool_returned_str, [], {"success": False, "num_images": 0}

    tool_stat = {"success": True, "num_images": len(tool_returned_images)}
    print(f"[Image Search] Done, {len(tool_returned_images)} images returned.")
    return tool_returned_str, tool_returned_images, tool_stat
