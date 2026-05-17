from corpus.db import get_connection

conn = None
cur = None
try:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    cur.fetchone()
    print("Database connection successful")
except Exception as e:
    print(f"Database error ({type(e).__name__}): {e}")
finally:
    if cur is not None:
        cur.close()
    if conn is not None:
        conn.close()
