import importlib
from types import SimpleNamespace
import urllib.request

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class FakeModelInfoResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return b'{"capabilities":["tools"]}'


class FakeStreamingModel:
    def __init__(self):
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "calculator",
                        "args": '{"expression":"2 + 2"}',
                        "id": "calculator-call",
                        "index": 0,
                    }
                ],
            )
            return

        assert any(
            getattr(message, "type", "") == "tool"
            and "2 + 2 = 4" in message.content
            for message in messages
        )
        yield AIMessageChunk(content="The answer is ")
        yield AIMessageChunk(content="4.")


def test_agent_streams_final_tokens_after_non_purchase_tool(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeModelInfoResponse(),
    )
    backend = importlib.import_module("backend")
    fake_model = FakeStreamingModel()
    monkeypatch.setattr(backend, "tool_calling_supported", True)
    monkeypatch.setattr(backend, "tool_llm", fake_model)

    graph = StateGraph(backend.ChatState)
    graph.add_node("agent", backend.chat_node)
    graph.add_node("tools", backend.run_tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", backend.route_after_agent)
    graph.add_edge("tools", "agent")

    events = list(
        graph.compile().stream(
            {
                "messages": [HumanMessage(content="What is 2 + 2?")],
                "tool_iterations": 0,
            },
            stream_mode="custom",
            config={"recursion_limit": 12},
        )
    )
    assert events == [
        {"token": "The answer is "},
        {"token": "4."},
    ]


@pytest.mark.parametrize(
    ("approved", "expected_tool_calls", "expected_result"),
    [
        (False, 0, "The user rejected this stock purchase."),
        (True, 1, "Paper order accepted."),
    ],
)
def test_stock_purchase_runs_only_after_approval(
    monkeypatch,
    approved,
    expected_tool_calls,
    expected_result,
):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeModelInfoResponse(),
    )
    backend = importlib.import_module("backend")

    class FakeApprovalModel:
        def __init__(self):
            self.calls = 0

        def stream(self, messages):
            self.calls += 1
            if self.calls == 1:
                yield AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "purchase_stock",
                            "args": '{"symbol":"AAPL","quantity":2}',
                            "id": "purchase-call",
                            "index": 0,
                        }
                    ],
                )
                return
            assert any(
                isinstance(message, ToolMessage)
                and message.content == expected_result
                for message in messages
            )
            yield AIMessageChunk(content="Purchase flow completed.")

    class RecordingToolNode:
        def __init__(self):
            self.calls = 0

        def invoke(self, state):
            self.calls += 1
            return {
                "messages": [
                    ToolMessage(
                        content="Paper order accepted.",
                        tool_call_id="purchase-call",
                    )
                ]
            }

    fake_model = FakeApprovalModel()
    recording_tool_node = RecordingToolNode()
    monkeypatch.setattr(backend, "tool_calling_supported", True)
    monkeypatch.setattr(backend, "tool_llm", fake_model)
    monkeypatch.setattr(backend, "tool_node", recording_tool_node)

    graph = StateGraph(backend.ChatState)
    graph.add_node("agent", backend.chat_node)
    graph.add_node("tools", backend.run_tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", backend.route_after_agent)
    graph.add_edge("tools", "agent")
    compiled = graph.compile(checkpointer=InMemorySaver())
    config = {
        "configurable": {"thread_id": f"stock-purchase-{approved}"},
        "recursion_limit": 12,
    }
    pending_events = list(
        compiled.stream(
            {
                "messages": [HumanMessage(content="What is 2 + 2?")],
                "tool_iterations": 0,
            },
            stream_mode="custom",
            config=config,
        )
    )
    assert pending_events == []
    assert compiled.get_state(config).tasks[0].interrupts[0].value == {
        "type": "purchase_approval",
        "tool_calls": [
            {
                "id": "purchase-call",
                "name": "purchase_stock",
                "args": {"symbol": "AAPL", "quantity": 2},
            }
        ],
    }

    events = list(
        compiled.stream(
            Command(resume={"approved": approved}),
            stream_mode="custom",
            config=config,
        )
    )

    assert recording_tool_node.calls == expected_tool_calls
    assert events == [{"token": "Purchase flow completed."}]


def test_rejecting_purchase_does_not_block_other_tools(monkeypatch):
    backend = importlib.import_module("backend")
    tool_node_calls = []

    class RecordingToolNode:
        def invoke(self, state):
            tool_calls = state["messages"][-1].tool_calls
            tool_node_calls.append([call["name"] for call in tool_calls])
            return {
                "messages": [
                    ToolMessage(
                        content="2 + 2 = 4",
                        tool_call_id=tool_calls[0]["id"],
                    )
                ]
            }

    def reject_purchase(payload):
        assert payload["type"] == "purchase_approval"
        assert [call["name"] for call in payload["tool_calls"]] == [
            "purchase_stock"
        ]
        return {"approved": False}

    monkeypatch.setattr(backend, "interrupt", reject_purchase)
    monkeypatch.setattr(backend, "tool_node", RecordingToolNode())
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "2 + 2"},
                "id": "calculator-call",
                "type": "tool_call",
            },
            {
                "name": "purchase_stock",
                "args": {"symbol": "AAPL", "quantity": 2},
                "id": "purchase-call",
                "type": "tool_call",
            },
        ],
    )

    result = backend.run_tools_node(
        {"messages": [message], "tool_iterations": 0}
    )

    assert tool_node_calls == [["calculator"]]
    assert [item.tool_call_id for item in result["messages"]] == [
        "calculator-call",
        "purchase-call",
    ]


def test_react_fallback_returns_tool_call_for_approval(monkeypatch):
    backend = importlib.import_module("backend")

    class FakeReActModel:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "Thought: calculate the result\n"
                    "Action: calculator\n"
                    'Action Input: {"expression":"2 + 2"}'
                )
            )

    monkeypatch.setattr(backend, "llm", FakeReActModel())

    class ToolThatMustNotRun:
        args = {"expression": {}}

        def invoke(self, *_args, **_kwargs):
            pytest.fail("Tool ran before approval")

    monkeypatch.setitem(
        backend.tools_by_name,
        "calculator",
        ToolThatMustNotRun(),
    )

    response = backend._run_react_agent([HumanMessage(content="What is 2 + 2?")])

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "calculator"
    assert response.tool_calls[0]["args"] == {"expression": "2 + 2"}
    assert response.tool_calls[0]["id"]


def test_frontend_forwards_only_streamed_tokens(monkeypatch):
    import frontend

    class FakeGraph:
        def stream(self, state, *, config, stream_mode):
            assert state["messages"][0].content == "Hello"
            assert stream_mode == "custom"
            yield {"token": "Hi "}
            yield {"status": "Using a tool"}
            yield {"token": "there"}

    monkeypatch.setattr(frontend, "chatbot", FakeGraph())

    assert list(frontend._stream_chat_response("Hello", {})) == ["Hi ", "there"]


def test_frontend_extracts_chat_input_text_and_files():
    import frontend

    uploaded_file = object()

    assert frontend._chat_input_parts("Hello") == ("Hello", [])
    assert frontend._chat_input_parts(
        SimpleNamespace(text="Summarize this", files=[uploaded_file])
    ) == ("Summarize this", [uploaded_file])
