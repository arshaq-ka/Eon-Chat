from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

from tools.common import trim_output


@tool
def get_current_datetime(timezone: str = "Asia/Kolkata") -> str:
    """Get the current date and time in a timezone."""
    try:
        now = datetime.now(ZoneInfo(timezone))
        return trim_output(
            f"{now.strftime('%A, %Y-%m-%d')} at {now.strftime('%H:%M:%S')} "
            f"({timezone})"
        )
    except (ZoneInfoNotFoundError, ValueError) as exc:
        return f"Date/time lookup failed: unknown timezone '{timezone}' ({exc})."
    except Exception as exc:
        return trim_output(f"Date/time lookup failed: {exc}")
