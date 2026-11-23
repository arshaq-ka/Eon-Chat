import os

import feedparser
import requests
from langchain_core.tools import tool

from tools.common import trim_output


@tool
def get_current_news(query: str, max_results: int = 5) -> str:
    """Find recent headlines for a topic."""
    try:
        max_results = max(1, min(int(max_results), 10))
        params = {"q": query, "hl": "en", "gl": "US", "ceid": "US:en"}
        api_key = os.getenv("GNEWS_API_KEY")
        if api_key:
            response = requests.get(
                "https://gnews.io/api/v4/search",
                params={
                    "q": query,
                    "max": max_results,
                    "lang": "en",
                    "apikey": api_key,
                },
                timeout=10,
            )
            response.raise_for_status()
            articles = response.json().get("articles", [])
            lines = [
                f"{item.get('title', 'Untitled')} | "
                f"{item.get('source', {}).get('name', 'Unknown source')} | "
                f"{item.get('publishedAt', 'Unknown date')} | "
                f"{item.get('url', '')}"
                for item in articles[:max_results]
            ]
        else:
            response = requests.get(
                "https://news.google.com/rss/search",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            lines = []
            for entry in feed.entries[:max_results]:
                source = entry.get("source", {}).get("title", "Unknown source")
                lines.append(
                    f"{entry.get('title', 'Untitled')} | {source} | "
                    f"{entry.get('published', 'Unknown date')} | "
                    f"{entry.get('link', '')}"
                )
        return trim_output("\n".join(lines) or "No news results found.")
    except Exception:
        return "News search failed. Check the connection and news API configuration."
