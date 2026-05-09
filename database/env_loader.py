"""Load repo-root `.env` (parent of the `database/` directory)."""

from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_repo_dotenv() -> None:
    load_dotenv(repo_root() / ".env")
