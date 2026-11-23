MAX_TOOL_OUTPUT = 1500


def trim_output(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[: MAX_TOOL_OUTPUT - 3] + "..."
