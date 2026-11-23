from tools.calculator import calculator
from tools.current_datetime import get_current_datetime
from tools.current_news import get_current_news
from tools.document_search import search_uploaded_documents
from tools.weather import get_weather
from tools.web_search import web_search


TOOLS = [
    get_current_news,
    get_weather,
    calculator,
    get_current_datetime,
    search_uploaded_documents,
    web_search,
]

__all__ = [
    "TOOLS",
    "calculator",
    "get_current_datetime",
    "get_current_news",
    "get_weather",
    "search_uploaded_documents",
    "web_search",
]
