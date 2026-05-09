import os
import sys
from pathlib import Path

import psycopg2

_db_dir = Path(__file__).resolve().parent
if str(_db_dir) not in sys.path:
    sys.path.insert(0, str(_db_dir))

from env_loader import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        database=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        port=os.environ.get("PGPORT", 5432),
    )
