import importlib
from datetime import datetime
from zoneinfo import ZoneInfo

from tools import (
    calculator,
    get_current_datetime,
    get_current_news,
    get_weather,
    web_search,
)


class FakeResponse:
    def __init__(self, *, content=b"", data=None):
        self.content = content
        self.data = data or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


def test_current_news_uses_rss(monkeypatch):
    rss = b"""<?xml version="1.0"?>
    <rss version="2.0"><channel><item>
    <title>New AI research</title><link>https://example.com/news</link>
    <pubDate>Mon, 23 Nov 2026 10:00:00 GMT</pubDate>
    <source url="https://example.com">Example News</source>
    </item></channel></rss>"""
    monkeypatch.delenv("GNEWS_API_KEY", raising=False)
    monkeypatch.setattr(
        "tools.current_news.requests.get",
        lambda *args, **kwargs: FakeResponse(content=rss),
    )

    result = get_current_news.invoke({"query": "AI", "max_results": 1})

    assert "New AI research" in result
    assert "Example News" in result
    assert "https://example.com/news" in result


def test_weather_geocodes_and_returns_forecast(monkeypatch):
    responses = iter(
        [
            FakeResponse(
                data={
                    "results": [
                        {
                            "name": "Mangalore",
                            "country": "India",
                            "latitude": 12.9,
                            "longitude": 74.8,
                        }
                    ]
                }
            ),
            FakeResponse(
                data={
                    "current": {
                        "temperature_2m": 29,
                        "relative_humidity_2m": 70,
                        "precipitation": 0,
                        "weather_code": 2,
                        "wind_speed_10m": 12,
                    },
                    "daily": {
                        "time": ["2026-11-23"],
                        "precipitation_probability_max": [20],
                        "temperature_2m_max": [31],
                        "temperature_2m_min": [25],
                        "weather_code": [2],
                    },
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "tools.weather.requests.get",
        lambda *args, **kwargs: next(responses),
    )

    result = get_weather.invoke({"city": "Mangalore", "days": 1})

    assert "Mangalore, India" in result
    assert "29 °C" in result
    assert "rain chance 20%" in result


def test_calculator_basic_and_functions():
    assert "810" in calculator.invoke({"expression": "4500 * 18 / 100"})
    assert "3.0" in calculator.invoke({"expression": "sqrt(9)"})


def test_calculator_rejects_division_by_zero_and_code():
    assert "Error:" in calculator.invoke({"expression": "1 / 0"})
    assert "Error:" in calculator.invoke(
        {"expression": "__import__('os').system('whoami')"}
    )


def test_current_datetime_returns_requested_zone():
    result = get_current_datetime.invoke({"timezone": "Asia/Kolkata"})
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    assert now.strftime("%Y-%m-%d") in result
    assert "Asia/Kolkata" in result


def test_web_search_returns_top_results(monkeypatch):
    class FakeDDGS:
        def __init__(self, timeout):
            assert timeout == 10

        def text(self, query, max_results):
            assert query == "Python"
            assert max_results == 5
            return [
                {
                    "title": "Python",
                    "body": "Python website",
                    "href": "https://python.org",
                }
            ]

    web_search_module = importlib.import_module("tools.web_search")
    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)

    result = web_search.invoke({"query": "Python"})

    assert "Python website" in result
    assert "https://python.org" in result
