import os
import re
import requests
from urllib.parse import quote_plus
from openai import OpenAI


# MiMo-2.5 API config (OpenAI compatible)
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "MiMo-2.5")


def _search_bing(query: str, topk: int = 5) -> list[dict]:
    """
    Free Bing search by scraping cn.bing.com.
    Returns list of dicts: [{"title": ..., "href": ..., "body": ...}, ...]
    """
    try:
        url = f"https://cn.bing.com/search?q={quote_plus(query)}&count={topk}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        results = []
        # Parse h2 > a for title and link
        h2_links = re.findall(r'<h2[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL)
        # Parse cite for snippet text
        cites = re.findall(r'<cite[^>]*>(.*?)</cite>', html, re.DOTALL)

        for i, (href, title_html) in enumerate(h2_links[:topk]):
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            snippet = ""
            if i < len(cites):
                snippet = re.sub(r'<[^>]+>', '', cites[i]).strip()
            results.append({"title": title, "href": href, "body": snippet})

        return results
    except Exception as e:
        print(f"[Text Search] Bing search error: {e}")
        return []


def _fetch_page_content(url: str, max_chars: int = 8000) -> str:
    """
    Fetch and extract main text content from a URL using trafilatura.
    """
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text[:max_chars]
    except Exception as e:
        print(f"[Text Search] trafilatura error for {url}: {e}")

    # fallback: requests + basic extraction
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


def _summarize_with_mimo(query: str, passages: list[str]) -> list[str]:
    """
    Use MiMo-2.5 to summarize each passage given the original query.
    Returns list of summary strings, one per passage.
    """
    if not MIMO_API_KEY or not MIMO_BASE_URL:
        # No API configured, return raw passages
        return passages

    client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
    summaries = []
    for passage in passages:
        try:
            resp = client.chat.completions.create(
                model=MIMO_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise information extractor. Given a user query and a webpage passage, extract and summarize the information relevant to the query. Return only the relevant facts, no preamble."},
                    {"role": "user", "content": f"Query: {query}\n\nPassage:\n{passage[:6000]}\n\nSummarize the information relevant to the query in 2-3 sentences."}
                ],
                max_tokens=256,
                temperature=0.3,
            )
            summaries.append(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"[Text Search] MiMo summarization error: {e}")
            summaries.append(passage[:500])  # fallback to raw text
    return summaries


def call_text_search(text_query: str, data_id: str = None):
    """
    Text search tool: Bing search + webpage extraction + MiMo summarization.

    Args:
        text_query: The model's search query.
        data_id: Optional data_id for cache lookup (unused for text search).

    Returns:
        tool_returned_str (str): Formatted search results string.
        tool_stat (dict): Tool status.
    """
    print(f"[Text Search] Query: {text_query}")

    # Step 1: Search via DuckDuckGo (Bing-like)
    search_results = _search_bing(text_query, topk=5)
    if not search_results:
        tool_returned_str = "[Text Search Results] There is an error encountered in performing search. Please reason with your own capabilities."
        return tool_returned_str, {"success": False, "num_results": 0}

    # Step 2: Fetch and extract content from top pages
    raw_passages = []
    page_titles = []
    page_links = []
    for result in search_results[:3]:  # top 3 to keep latency manageable
        url = result.get("href", "")
        title = result.get("title", "")
        # Try using DuckDuckGo's snippet first, then fetch full page
        snippet = result.get("body", "")
        content = _fetch_page_content(url)
        passage = content if content else snippet
        if passage:
            raw_passages.append(passage)
            page_titles.append(title)
            page_links.append(url)

    if not raw_passages:
        tool_returned_str = "[Text Search Results] There is an error encountered in performing search. Please reason with your own capabilities."
        return tool_returned_str, {"success": False, "num_results": 0}

    # Step 3: Summarize with MiMo-2.5
    summaries = _summarize_with_mimo(text_query, raw_passages)

    # Step 4: Format output
    tool_returned_str = "[Text Search Results] Below are the text summaries of the most relevant webpages related to your query, ranked in descending order of relevance:\n"
    for i, (title, link, summary) in enumerate(zip(page_titles, page_links, summaries)):
        tool_returned_str += f"{i+1}. (webpage link) {title}\nSummary: {summary}\n\n"

    tool_stat = {"success": True, "num_results": len(summaries)}
    print(f"[Text Search] Done, {len(summaries)} results returned.")
    return tool_returned_str, tool_stat
