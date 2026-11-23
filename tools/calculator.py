import ast
import math
import operator

from langchain_core.tools import tool

from tools.common import trim_output


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _calculate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = _calculate(node.left)
        right = _calculate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent magnitude must be 100 or less.")
        return BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](_calculate(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        args = [_calculate(argument) for argument in node.args]
        if node.func.id == "log" and len(args) in (1, 2):
            return math.log(*args)
        if node.func.id != "log" and len(args) == 1:
            return FUNCTIONS[node.func.id](*args)
    raise ValueError("Only basic arithmetic and sqrt/log/sin/cos/tan are allowed.")


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression."""
    try:
        result = _calculate(ast.parse(expression, mode="eval"))
        if not math.isfinite(result):
            return "Error: the result is not a finite number."
        return trim_output(f"{expression} = {result}")
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return trim_output(f"Error: invalid expression: {exc}")
