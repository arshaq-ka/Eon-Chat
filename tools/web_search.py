from ddgs import DDGS
from langchain_core.tools import tool

from tools.common import trim_output


@tool
def web_search(query: str) -> str:
    """Search the web and return the top five results."""
    try:
        results = DDGS(timeout=10).text(
            query,
            max_results=5,
        )
        lines = [
            f"{item.get('title', 'Untitled')} | "
            f"{item.get('body', '')} | {item.get('href', '')}"
            for item in results
        ]
        return trim_output("\n".join(lines) or "No web results found.")
    except Exception as exc:
        return trim_output(f"Web search failed: {exc}")
