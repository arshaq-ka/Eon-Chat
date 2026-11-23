import requests
from langchain_core.tools import tool

from tools.common import trim_output


WEATHER_CODES = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


@tool
def get_weather(city: str, days: int = 1) -> str:
    """Get current weather and a short forecast for a city."""
    try:
        days = max(1, min(int(days), 7))
        geocode_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geocode_response.raise_for_status()
        locations = geocode_response.json().get("results", [])
        if not locations:
            return f"Weather unavailable: couldn't find the city '{city}'."

        location = locations[0]
        forecast_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,precipitation,"
                    "weather_code,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "forecast_days": days,
                "timezone": "auto",
            },
            timeout=10,
        )
        forecast_response.raise_for_status()
        data = forecast_response.json()
        current = data["current"]
        lines = [
            f"{location['name']}, {location.get('country', '')}: "
            f"{current['temperature_2m']} °C, "
            f"{WEATHER_CODES.get(current['weather_code'], 'Unknown conditions')}; "
            f"humidity {current['relative_humidity_2m']}%; "
            f"wind {current['wind_speed_10m']} km/h; "
            f"rain now {current['precipitation']} mm."
        ]
        daily = data.get("daily", {})
        for index, date in enumerate(daily.get("time", [])[:days]):
            rain_chance = daily.get("precipitation_probability_max", [])
            high = daily.get("temperature_2m_max", [])
            low = daily.get("temperature_2m_min", [])
            code = daily.get("weather_code", [])
            lines.append(
                f"{date}: {WEATHER_CODES.get(code[index], 'Unknown conditions')}; "
                f"{low[index]}–{high[index]} °C; "
                f"rain chance {rain_chance[index]}%."
            )
        return trim_output("\n".join(lines))
    except Exception as exc:
        return trim_output(f"Weather lookup failed: {exc}")
