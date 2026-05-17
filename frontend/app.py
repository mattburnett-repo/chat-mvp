"""Streamlit UI for Chat MVP — calls the FastAPI `/query` endpoint.

Run from repo root: python3.13.12 -m streamlit run frontend/app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
import utils.env_loader  # noqa: F401


def api_base() -> str:
    return os.environ.get("CHAT_MVP_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def default_top_k() -> int:
    return int(os.environ.get("QUERY_TOP_K", "5"))


def query_api(
    question: str, top_k: int, session_id: str | None
) -> tuple[str, list[dict], str]:
    body: dict = {"query": question, "top_k": top_k}
    if session_id:
        body["session_id"] = session_id
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{api_base()}/query", json=body)
    r.raise_for_status()
    data = r.json()
    answer = data.get("answer", "")
    sources = data.get("sources") or []
    sid = data.get("session_id") or ""
    return answer, sources, sid


st.set_page_config(page_title="Chat MVP", page_icon="💬", layout="centered")
st.title("CIB Mango Tree | Chat MVP")

with st.sidebar:
    st.caption("RAG chat using the FastAPI backend and ingested documents.")
    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()
    top_k = st.number_input(
        "Chunks to retrieve (top_k)",
        min_value=1,
        max_value=20,
        value=min(max(default_top_k(), 1), 20),
        step=1,
    )
    st.caption(f"API base URL: `{api_base()}`")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    url = s.get("source_url", "")
                    idx = s.get("chunk_index", "")
                    st.markdown(f"- [{url}]({url}) (chunk {idx})")

if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Querying…"):
                answer, sources, sid = query_api(
                    prompt, int(top_k), st.session_state.conversation_id
                )
                if sid:
                    st.session_state.conversation_id = sid
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        url = s.get("source_url", "")
                        idx = s.get("chunk_index", "")
                        st.markdown(f"- [{url}]({url}) (chunk {idx})")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except httpx.HTTPStatusError as e:
            detail = e.response.text
            try:
                body = e.response.json()
                detail = body.get("detail", detail)
            except Exception:
                pass
            err = f"**API error ({e.response.status_code})** — {detail}"
            st.markdown(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "sources": []}
            )
        except httpx.RequestError as e:
            err = (
                f"**Could not reach the API** at `{api_base()}`. "
                f"Start the backend (e.g. `python3.13.12 -m uvicorn backend.main:app`) "
                f"and try again. ({e})"
            )
            st.markdown(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "sources": []}
            )
