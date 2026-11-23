import importlib
import urllib.request

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.graph import END, START, StateGraph


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


def test_agent_streams_final_tokens_after_tool_result(monkeypatch):
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
