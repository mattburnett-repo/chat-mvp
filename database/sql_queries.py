SELECT_DOCUMENTS_WITHOUT_EMBEDDING = """
    SELECT id, content
    FROM documents
    WHERE embedding IS NULL
"""

UPDATE_DOCUMENT_EMBEDDING = """
    UPDATE documents SET embedding = %s WHERE id = %s
"""

DELETE_DOCUMENTS_FOR_SOURCE_URL = """
    DELETE FROM documents WHERE source_url = %s
"""

INSERT_DOCUMENT_CHUNK = """
    INSERT INTO documents (source_url, chunk_index, title, content, fetched_at, metadata)
    VALUES (%s, %s, %s, %s, NOW(), %s)
"""

SELECT_DOCUMENTS_BY_VECTOR_SIMILARITY = """
    SELECT source_url, chunk_index, content
    FROM documents
    WHERE embedding IS NOT NULL
    ORDER BY embedding <-> %s::vector
    LIMIT %s
"""

INSERT_CONVERSATION_IF_ABSENT = """
    INSERT INTO conversations (session_id)
    VALUES (%s)
    ON CONFLICT (session_id) DO NOTHING
"""

SELECT_CHAT_MESSAGES_FOR_SESSION = """
    SELECT message
    FROM chat_messages
    WHERE session_id = %s
    ORDER BY created_at ASC, id ASC
"""

INSERT_CHAT_MESSAGE = """
    INSERT INTO chat_messages (session_id, message)
    VALUES (%s, %s)
"""

DELETE_CHAT_MESSAGES_FOR_SESSION = """
    DELETE FROM chat_messages
    WHERE session_id = %s
"""
