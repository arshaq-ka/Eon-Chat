from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

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


def _upload_fingerprint(files) -> str:
    digest = sha256()
    for uploaded_file in files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def _stream_chat_response(prompt: str, config: dict):
    for event in chatbot.stream(
        {
            "messages": [HumanMessage(content=prompt)],
            "tool_iterations": 0,
        },
        config=config,
        stream_mode="custom",
    ):
        if isinstance(event, dict) and isinstance(event.get("token"), str):
            yield event["token"]


@st.fragment(run_every="1s")
def render_index_status() -> None:
    future = st.session_state.get("rag_index_future")
    if future is not None and future.done():
        try:
            st.session_state["rag_index_statuses"] = future.result()
        except Exception as exc:
            st.session_state["rag_index_statuses"] = [
                f"Document indexing failed: {exc}"
            ]
        st.session_state["rag_index_future"] = None
        st.rerun(scope="app")

    if future is not None:
        file_count = st.session_state.get("rag_index_file_count", 1)
        st.info(
            f"Indexing {file_count} document(s) in the background. "
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
        [data-testid="stMainBlockContainer"] {
            max-width: 900px;
            padding-top: 1.5rem;
            padding-bottom: 7rem;
        }
        .eon-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 0.2rem 0 1.5rem;
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

        st.markdown("#### Your documents")
        active_future = st.session_state.get("rag_index_future")
        is_indexing = active_future is not None and not active_future.done()
        uploaded_files = st.file_uploader(
            "Add PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="rag_document_uploads",
            disabled=is_indexing,
            label_visibility="collapsed",
        )
        if uploaded_files:
            fingerprint = _upload_fingerprint(uploaded_files)
            if fingerprint != st.session_state.get("rag_upload_fingerprint"):
                st.session_state["rag_upload_fingerprint"] = fingerprint
                st.session_state["rag_index_file_count"] = len(uploaded_files)
                st.session_state["rag_index_statuses"] = []
                file_payloads = [
                    (uploaded_file.name, uploaded_file.getvalue())
                    for uploaded_file in uploaded_files
                ]
                st.session_state["rag_index_future"] = INDEX_EXECUTOR.submit(
                    index_uploaded_documents,
                    file_payloads,
                )
                st.rerun()
        elif not is_indexing:
            st.session_state["rag_upload_fingerprint"] = None

        render_index_status()

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

    prompt = st.chat_input("Message EON")
    if suggested_prompt:
        prompt = suggested_prompt

    if prompt:
        remember_user_name(prompt)
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            response = st.write_stream(
                _stream_chat_response(prompt, thread_config)
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    run_streamlit_app()
