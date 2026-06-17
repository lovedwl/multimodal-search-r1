import os
import re
import json
import hashlib
import requests
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


# MiMo-2.5 API config (OpenAI compatible)
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "MiMo-2.5")

# Text search query cache (disk-backed)
_TEXT_CACHE_PATH = os.environ.get(
    "TEXT_SEARCH_CACHE_PATH",
    os.path.join(os.path.dirname(__file__), "../../../data/FVQA/text_search_cache.json"),
)
_text_cache = None


def _load_text_cache() -> dict:
    """Load text search cache from disk."""
    global _text_cache
    if _text_cache is not None:
        return _text_cache
    path = os.path.abspath(_TEXT_CACHE_PATH)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _text_cache = json.load(f)
            print(f"[Text Search] Loaded query cache: {len(_text_cache)} entries")
        except Exception as e:
            print(f"[Text Search] Failed to load query cache: {e}")
            _text_cache = {}
    else:
        _text_cache = {}
    return _text_cache


def _save_text_cache():
    """Save text search cache to disk."""
    if _text_cache is None:
        return
    path = os.path.abspath(_TEXT_CACHE_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_text_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Text Search] Failed to save query cache: {e}")


def _normalize_query(query: str) -> str:
    """Normalize query for cache key: lowercase, strip, collapse whitespace."""
    return " ".join(query.lower().strip().split())


def _cache_key(query: str) -> str:
    """Generate cache key from normalized query."""
    normalized = _normalize_query(query)
    return hashlib.md5(normalized.encode()).hexdigest()


def _search_bing(query: str, topk: int = 5) -> list[dict]:
    """
    Free Bing search by scraping bing.com.
    Returns list of dicts: [{"title": ..., "href": ..., "body": ...}, ...]
    """
    try:
        url = f"https://www.bing.com/search?q={quote_plus(query)}&count={topk}&cc=us&setlang=en&mkt=en-US&ensearch=1"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        results = []
        h2_links = re.findall(r'<h2[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL)
        cites = re.findall(r'<cite[^>]*>(.*?)</cite>', html, re.DOTALL)

        # Chinese domains to deprioritize
        cn_domains = ('baike.baidu.com', 'zhuanlan.zhihu.com', 'zhihu.com', 'bilibili.com', 'weibo.com', 'douban.com')

        en_results = []
        cn_results = []
        for i, (href, title_html) in enumerate(h2_links):
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            snippet = ""
            # Extract actual URL from cite tag
            real_url = href  # fallback to redirect URL
            if i < len(cites):
                snippet = re.sub(r'<[^>]+>', '', cites[i]).strip()
                # cite format: "https://domain.com › path › to › page"
                cite_url_match = re.match(r'(https?://[^\s›]+)', snippet)
                if cite_url_match:
                    real_url = cite_url_match.group(1)
            entry = {"title": title, "href": real_url, "body": snippet}
            if any(d in real_url for d in cn_domains):
                cn_results.append(entry)
            else:
                en_results.append(entry)

        results = en_results + cn_results
        return results
    except Exception as e:
        print(f"[Text Search] Bing search error: {e}")
        return []


def _fetch_page_content(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract main text content from a URL using trafilatura."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text[:max_chars]
    except Exception as e:
        print(f"[Text Search] trafilatura error for {url}: {e}")

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        import trafilatura
        text = trafilatura.extract(resp.text)
        if text:
            return text[:max_chars]
    except Exception as e:
        print(f"[Text Search] fallback fetch error for {url}: {e}")

    return ""


def _fetch_pages_parallel(results: list[dict], max_workers: int = 3) -> list[tuple[str, str, str]]:
    """Fetch page content in parallel."""
    def _fetch_one(item):
        title = item["title"]
        link = item["href"]
        snippet = item.get("body", "")
        content = _fetch_page_content(link)
        if not content:
            content = snippet
        if content:
            return (title, link, content)
        return None

    fetched = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_one, r) for r in results[:3]]
        for future in as_completed(futures):
            result = future.result()
            if result:
                fetched.append(result)
    return fetched


def _summarize_with_mimo(query: str, passages: list[tuple[str, str, str]]) -> list[str]:
    """Use MiMo-2.5 to summarize passages (merged into one call)."""
    if not MIMO_API_KEY or not MIMO_BASE_URL:
        return [content[:500] for _, _, content in passages]

    passage_text = ""
    for i, (title, link, content) in enumerate(passages):
        passage_text += f"\n--- Passage {i+1}: {title} (source: {link}) ---\n{content[:3000]}\n"

    client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise information extractor. Given a user query and multiple webpage passages, extract and summarize the information relevant to the query from EACH passage separately. Return a numbered list of summaries, one per passage."},
                {"role": "user", "content": f"Query: {query}\n\nPassages:{passage_text}\n\nProvide a numbered list of summaries (one per passage). Each summary should be 1-3 sentences. If a passage has no relevant information, write 'No relevant information.'"}
            ],
            max_tokens=512,
            temperature=0.3,
        )
        response_text = resp.choices[0].message.content.strip()

        summaries = re.split(r'\d+\.\s*', response_text)
        summaries = [s.strip() for s in summaries if s.strip()]

        while len(summaries) < len(passages):
            summaries.append("No relevant information.")

        return summaries[:len(passages)]
    except Exception as e:
        print(f"[Text Search] MiMo summarization error: {e}")
        return [content[:500] for _, _, content in passages]


def call_text_search(text_query: str, data_id: str = None):
    """
    Text search tool with query cache.
    Cache key is normalized query (lowercase, stripped). Exact match returns cached result instantly.

    Args:
        text_query: The model's search query.
        data_id: Optional data_id (unused).

    Returns:
        tool_returned_str (str): Formatted search results string.
        tool_stat (dict): Tool status.
    """
    print(f"[Text Search] Query: {text_query}")

    # Check cache
    cache = _load_text_cache()
    key = _cache_key(text_query)
    if key in cache:
        print(f"[Text Search] Cache hit!")
        return cache[key]["result_str"], cache[key]["stat"]

    # Step 1: Search via Bing
    search_results = _search_bing(text_query, topk=5)
    if not search_results:
        tool_returned_str = "[Text Search Results] There is an error encountered in performing search. Please reason with your own capabilities."
        tool_stat = {"success": False, "num_results": 0}
        return tool_returned_str, tool_stat

    # Step 2: Fetch pages in parallel
    fetched = _fetch_pages_parallel(search_results, max_workers=3)
    if not fetched:
        tool_returned_str = "[Text Search Results] There is an error encountered in performing search. Please reason with your own capabilities."
        tool_stat = {"success": False, "num_results": 0}
        return tool_returned_str, tool_stat

    # Step 3: Summarize with MiMo
    summaries = _summarize_with_mimo(text_query, fetched)

    # Step 4: Format output
    tool_returned_str = "[Text Search Results] Below are the text summaries of the most relevant webpages related to your query, ranked in descending order of relevance:\n"
    for i, ((title, link, _), summary) in enumerate(zip(fetched, summaries)):
        tool_returned_str += f"{i+1}. (webpage link) {title}\nSummary: {summary}\n\n"

    tool_stat = {"success": True, "num_results": len(fetched)}

    # Save to cache
    cache[key] = {"query": text_query, "result_str": tool_returned_str, "stat": tool_stat}
    _save_text_cache()

    print(f"[Text Search] Done, {len(fetched)} results returned.")
    return tool_returned_str, tool_stat
