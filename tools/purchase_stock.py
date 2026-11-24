import os
import re

import requests
from langchain_core.tools import tool

from tools.common import trim_output


ALPACA_PAPER_ORDERS_URL = "https://paper-api.alpaca.markets/v2/orders"
SYMBOL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9.-]{0,9}\Z")


@tool
def purchase_stock(symbol: str, quantity: int) -> str:
    """Submit a whole-share market buy order to the Alpaca paper-trading account.

    The user must approve the order before this tool is executed.
    """
    normalized_symbol = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(normalized_symbol):
        return "Stock order not submitted: provide a valid US stock ticker symbol."
    if quantity <= 0:
        return "Stock order not submitted: quantity must be a positive whole number."

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return (
            "Stock order not submitted: set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY for an Alpaca paper-trading account."
        )

    try:
        response = requests.post(
            ALPACA_PAPER_ORDERS_URL,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
            json={
                "symbol": normalized_symbol,
                "qty": quantity,
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
            },
            timeout=15,
        )
    except requests.RequestException as exc:
        return trim_output(
            "Stock order submission could not be confirmed. Check the Alpaca "
            f"paper account's order activity before retrying. Details: {exc}"
        )

    if not response.ok:
        try:
            details = response.json()
        except ValueError:
            details = response.text
        return trim_output(
            f"Alpaca paper order rejected (HTTP {response.status_code}): {details}"
        )

    try:
        order = response.json()
    except ValueError:
        return (
            "Alpaca accepted the request but returned an unreadable response. "
            "Check the paper account's order activity before retrying."
        )
    if not isinstance(order, dict) or not order.get("id"):
        return (
            "Alpaca returned no order ID. Check the paper account's order "
            "activity before retrying."
        )

    return trim_output(
        f"Alpaca paper market buy order submitted: {quantity} share(s) of "
        f"{normalized_symbol}; order ID {order['id']}; "
        f"status {order.get('status', 'unknown')}. This is not a confirmation "
        "that the order has filled."
    )
