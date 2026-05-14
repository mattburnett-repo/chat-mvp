"""Postgres-backed LangChain `BaseChatMessageHistory` (JSONB rows in `chat_messages`)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
    trim_messages,
)
from psycopg2.extras import Json

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from corpus.db import get_connection  # noqa: E402
from corpus.sql_queries import (  # noqa: E402
    DELETE_CHAT_MESSAGES_FOR_SESSION,
    INSERT_CHAT_MESSAGE,
    SELECT_CHAT_MESSAGES_FOR_SESSION,
)
from utils.env_loader import load_repo_dotenv  # noqa: E402

load_repo_dotenv()


class PostgresChatMessageHistory(BaseChatMessageHistory):
    """Load ordered messages for a session; trim by token budget before model use."""

    def __init__(self, session_id: str, *, max_history_tokens: int) -> None:
        self._session_id = session_id
        self._max_history_tokens = max_history_tokens

    @property
    def messages(self) -> list[BaseMessage]:  # pyright: ignore[reportIncompatibleVariableOverride]
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(SELECT_CHAT_MESSAGES_FOR_SESSION, (self._session_id,))
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()

        raw = [row[0] for row in rows]
        loaded = messages_from_dict(raw)
        if not loaded:
            return []
        return trim_messages(
            loaded,
            max_tokens=self._max_history_tokens,
            token_counter="approximate",
            strategy="last",
            start_on="human",
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        conn = get_connection()
        try:
            cur = conn.cursor()
            for m in messages:
                cur.execute(
                    INSERT_CHAT_MESSAGE,
                    (self._session_id, Json(message_to_dict(m))),
                )
            conn.commit()
            cur.close()
        finally:
            conn.close()

    def clear(self) -> None:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(DELETE_CHAT_MESSAGES_FOR_SESSION, (self._session_id,))
            conn.commit()
            cur.close()
        finally:
            conn.close()
