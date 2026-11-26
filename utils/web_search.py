"""Small Tavily web-search client used for source-grounded AI responses."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse


_TRUSTED_DOMAINS = (
    "cancer.gov",
    "ninds.nih.gov",
    "mayoclinic.org",
    "nhs.uk",
    "pubmed.ncbi.nlm.nih.gov",
    "radiologyinfo.org",
    "cancer.org",
    "medlineplus.gov",
)


def is_web_search_available() -> bool:
    """Return whether a Tavily API key is configured."""
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search trusted medical sources and return normalized, citable results."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Tavily web search is not configured.")
    if not query.strip():
        return []

    payload = {
        "api_key": api_key,
        "query": query.strip(),
        "topic": "general",
        "search_depth": "basic",
        "max_results": max(1, min(max_results, 5)),
        "include_answer": False,
        "include_domains": list(_TRUSTED_DOMAINS),
    }
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("Tavily web search request failed.") from exc

    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise RuntimeError("Tavily returned an invalid search response.")

    results = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        try:
            parsed_url = urlparse(url)
        except ValueError:
            continue
        hostname = (parsed_url.hostname or "").lower()
        if (
            not title
            or not content
            or parsed_url.scheme != "https"
            or not any(hostname == domain or hostname.endswith(f".{domain}") for domain in _TRUSTED_DOMAINS)
        ):
            continue
        results.append({"title": title, "url": url, "content": content[:2500]})
        if len(results) >= payload["max_results"]:
            break
    return results
