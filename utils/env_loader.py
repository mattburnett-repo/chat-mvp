"""Load repo-root `.env` (project root is the parent of the `utils/` directory)."""

from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_repo_dotenv() -> None:
    load_dotenv(repo_root() / ".env")
