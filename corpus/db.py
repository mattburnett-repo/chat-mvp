import os
import sys
from pathlib import Path

import psycopg2

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from utils.env_loader import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        database=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        port=os.environ.get("PGPORT", 5432),
    )
