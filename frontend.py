from concurrent.futures import ThreadPoolExecutor
import json
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from backend import (
    MAX_TOOL_ITERATIONS,
    chatbot,
    get_saved_chats,
    remember_user_name,
)
from rag import index_uploaded_documents


INDEX_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="eon-document-indexer",
)


def select_prompt(prompt: str) -> None:
    st.session_state["suggested_prompt"] = prompt


def _chat_input_parts(value) -> tuple[str, list]:
    if isinstance(value, str):
        return value, []
    return value.text, value.files


def _queue_document_indexing(uploaded_files):
    file_payloads = [
        (uploaded_file.name, uploaded_file.getvalue())
        for uploaded_file in uploaded_files
    ]
    future = INDEX_EXECUTOR.submit(index_uploaded_documents, file_payloads)
    jobs = st.session_state.setdefault("rag_index_jobs", [])
    jobs.append((future, len(uploaded_files)))
    st.session_state["rag_index_statuses"] = []
    return future


def _collect_completed_index_jobs() -> None:
    pending_jobs = []
    statuses = []
    for future, file_count in st.session_state.get("rag_index_jobs", []):
        if future.done():
            try:
                statuses.extend(future.result())
            except Exception as exc:
                statuses.append(f"Failed to index {file_count} document(s): {exc}")
        else:
            pending_jobs.append((future, file_count))

    st.session_state["rag_index_jobs"] = pending_jobs
    if statuses:
        existing_statuses = st.session_state.get("rag_index_statuses", [])
        st.session_state["rag_index_statuses"] = existing_statuses + statuses


def _stream_chat_response(prompt: str | None, config: dict, resume=None):
    input_value = (
        Command(resume=resume)
        if resume is not None
        else {
            "messages": [HumanMessage(content=prompt)],
            "tool_iterations": 0,
        }
    )
    for event in chatbot.stream(
        input_value,
        config=config,
        stream_mode="custom",
    ):
        if isinstance(event, dict) and isinstance(event.get("token"), str):
            yield event["token"]


def _pending_tool_approval(config: dict):
    snapshot = chatbot.get_state(config)
    for task in snapshot.tasks:
        for pending_interrupt in task.interrupts:
            value = pending_interrupt.value
            if isinstance(value, dict) and value.get("type") == "purchase_approval":
                return value
    return None


def _render_tool_approval(config: dict) -> bool:
    pending = _pending_tool_approval(config)
    if pending is None:
        return False

    st.warning(
        "Review the Alpaca paper-trading purchase order(s). "
        "Approving submits the order(s) to the paper account."
    )
    for tool_call in pending.get("tool_calls", []):
        order = tool_call["args"]
        st.markdown(
            f"**Buy {order.get('quantity')} share(s) of "
            f"{order.get('symbol')} — market order**"
        )
        st.code(
            json.dumps(order, indent=2, ensure_ascii=False),
            language="json",
        )

    approve_column, reject_column = st.columns(2)
    with approve_column:
        if st.button(
            "Approve paper purchase(s)",
            type="primary",
            use_container_width=True,
        ):
            with st.chat_message("assistant", avatar="🤖"):
                response = st.write_stream(
                    _stream_chat_response(
                        None,
                        config,
                        resume={"approved": True},
                    )
                )
            if response:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            st.rerun()
    with reject_column:
        if st.button("Reject paper purchase(s)", use_container_width=True):
            with st.chat_message("assistant", avatar="🤖"):
                response = st.write_stream(
                    _stream_chat_response(
                        None,
                        config,
                        resume={"approved": False},
                    )
                )
            if response:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            st.rerun()

    return True


@st.fragment(run_every="1s")
def render_index_status() -> None:
    _collect_completed_index_jobs()
    jobs = st.session_state.get("rag_index_jobs", [])
    if jobs:
        file_count = sum(count for _future, count in jobs)
        st.info(
            f"Indexing or queueing {file_count} document(s) in the background. "
            "You can keep chatting."
        )
        return

    for status in st.session_state.get("rag_index_statuses", []):
        if status.startswith("Failed"):
            st.error(status)
        else:
            st.success(status)


def run_streamlit_app():
    st.set_page_config(
        page_title="EON Chat",
        page_icon="💬",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #212121;
        }
        [data-testid="stSidebar"] {
            background: #171717;
            border-right: 1px solid #303030;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 0 0.5rem 2rem;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 900px;
            padding-top: 1.5rem;
            padding-bottom: 7rem;
        }
        .eon-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 0.2rem 0 1.6rem;
            font-size: 1.1rem;
            font-weight: 650;
        }
        .eon-brand-mark {
            display: grid;
            width: 2rem;
            height: 2rem;
            place-items: center;
            border-radius: 0.7rem;
            background: #303030;
            color: #fff;
        }
        [data-testid="stSidebar"] .stButton > button {
            height: 3.1rem;
            min-height: 3.1rem;
            border: 1px solid #484743;
            border-radius: 0.8rem;
            background: #292826;
            color: #f5f5f5;
            font-weight: 500;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #686660;
            background: #302f2c;
        }
        .eon-welcome {
            max-width: 680px;
            margin: 14vh auto 2rem;
            text-align: center;
        }
        .eon-welcome-mark {
            display: grid;
            width: 3rem;
            height: 3rem;
            margin: 0 auto 1rem;
            place-items: center;
            border: 1px solid #444;
            border-radius: 1rem;
            background: #292929;
            font-size: 1.5rem;
        }
        .eon-welcome h1 {
            margin-bottom: 0.5rem;
            font-size: 2rem;
            font-weight: 600;
        }
        .eon-welcome p {
            color: #aaa;
            font-size: 1rem;
        }
        [data-testid="stChatMessage"] {
            padding: 1rem 1.1rem;
            border-radius: 1rem;
        }
        [data-testid="stChatInput"] {
            border-color: #444;
            border-radius: 1rem;
        }
        .stButton > button {
            border-radius: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    saved_chats = get_saved_chats()
    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = saved_chats[0][0] if saved_chats else uuid.uuid4().hex
            st.query_params["thread_id"] = thread_id
        st.session_state.thread_id = thread_id

    with st.sidebar:
        st.markdown(
            '<div class="eon-brand"><span class="eon-brand-mark">✦</span>'
            "<span>EON Chat</span></div>",
            unsafe_allow_html=True,
        )
        if st.button("＋  New chat", use_container_width=True):
            st.session_state.thread_id = uuid.uuid4().hex
            st.query_params["thread_id"] = st.session_state.thread_id
            st.rerun()

        if saved_chats:
            st.markdown("#### Recent chats")
            for saved_thread_id, title in saved_chats:
                if st.button(
                    title,
                    key=f"chat_{saved_thread_id}",
                    use_container_width=True,
                ):
                    st.session_state.thread_id = saved_thread_id
                    st.query_params["thread_id"] = saved_thread_id
                    st.rerun()

    thread_config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": (MAX_TOOL_ITERATIONS * 2) + 2,
    }
    if st.session_state.get("chat_history_thread_id") != st.session_state.thread_id:
        saved_state = chatbot.get_state(thread_config)
        messages = saved_state.values.get("messages", [])
        st.session_state.chat_history = [
            {
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content,
            }
            for message in messages
            if message.type in ("human", "ai")
            and (message.type == "human" or message.content)
        ]
        st.session_state.chat_history_thread_id = st.session_state.thread_id

    suggested_prompt = st.session_state.pop("suggested_prompt", None)
    if not st.session_state.chat_history and not suggested_prompt:
        st.markdown(
            """
            <div class="eon-welcome">
                <div class="eon-welcome-mark">✦</div>
                <h1>What can I help with?</h1>
                <p>Ask anything, or start with one of these ideas.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prompt_columns = st.columns(2)
        suggestions = [
            (
                "🧠  Explain a topic",
                "Break down a tricky idea",
                "Explain a difficult topic in simple terms, with a helpful example.",
            ),
            (
                "✍️  Help me write",
                "Draft something clear and polished",
                "Help me write a clear, friendly email about an important topic.",
            ),
            (
                "💡  Brainstorm ideas",
                "Find inspiration for a project",
                "Give me five creative ideas for a personal project.",
            ),
            (
                "📚  Find my next book",
                "Get a thoughtful recommendation",
                "Recommend a book and explain why I might enjoy reading it.",
            ),
        ]
        for index, (label, description, prompt_text) in enumerate(suggestions):
            with prompt_columns[index % 2]:
                with st.container(border=True):
                    st.button(
                        label,
                        key=f"suggestion_{index}",
                        use_container_width=True,
                        on_click=select_prompt,
                        args=(prompt_text,),
                    )
                    st.caption(description)

    for message in st.session_state.chat_history:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    approval_pending = _render_tool_approval(thread_config)
    render_index_status()
    chat_submission = st.chat_input(
        "Message EON",
        accept_file="multiple",
        file_type=["pdf", "docx"],
        max_upload_size=200,
        disabled=approval_pending,
    )
    if chat_submission is None:
        prompt = suggested_prompt or ""
        uploaded_files = []
    else:
        prompt, uploaded_files = _chat_input_parts(chat_submission)

    if uploaded_files:
        index_future = _queue_document_indexing(uploaded_files)
        if prompt.strip():
            with st.spinner("Indexing attachments before answering..."):
                index_future.result()
            _collect_completed_index_jobs()
        else:
            st.rerun()

    if prompt.strip():
        remember_user_name(prompt)
        attachment_names = [uploaded_file.name for uploaded_file in uploaded_files]
        display_prompt = prompt
        if attachment_names:
            display_prompt = (
                f"{prompt}\n\n📎 " + ", ".join(attachment_names)
            )
        st.session_state.chat_history.append(
            {"role": "user", "content": display_prompt}
        )
        with st.chat_message("user", avatar="🧑"):
            st.markdown(display_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            response = st.write_stream(
                _stream_chat_response(display_prompt, thread_config)
            )
        if response:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
        if _pending_tool_approval(thread_config) is not None:
            st.rerun()


if __name__ == "__main__":
    run_streamlit_app()
