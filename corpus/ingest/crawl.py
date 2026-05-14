"""
Ingest: (1) GitHub README + dev files linked from README (API + raw),
(2) each URL in `sources.SEED_URLS` once (no crawling, no link following).

Each source becomes one or more `documents` rows. Re-run replaces chunks per URL.

Run from repo root:  .venv/bin/python corpus/ingest/crawl.py

Requires: database env vars, `documents` with unique (source_url, chunk_index).

Env (`.env`): CRAWL_MAX_PAGES, CRAWL_REQUEST_TIMEOUT_S, CRAWL_DELAY_S, CRAWL_USER_AGENT,
CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from psycopg2.extras import Json

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chunking import chunk_text  # noqa: E402

from corpus.db import get_connection  # noqa: E402
from utils.env_loader import load_repo_dotenv  # noqa: E402
from corpus.sql_queries import (  # noqa: E402
    DELETE_DOCUMENTS_FOR_SOURCE_URL,
    INSERT_DOCUMENT_CHUNK,
)
from github_readme import iter_github_readme_docs  # noqa: E402
from sources import GITHUB_OWNER, GITHUB_REPO, SEED_URLS  # noqa: E402

load_repo_dotenv()

MAX_PAGES = int(os.environ["CRAWL_MAX_PAGES"].strip())
REQUEST_TIMEOUT_S = float(os.environ["CRAWL_REQUEST_TIMEOUT_S"].strip())
DELAY_S = float(os.environ["CRAWL_DELAY_S"].strip())
USER_AGENT = os.environ["CRAWL_USER_AGENT"].strip()


def canonical_url(url: str) -> str:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return ""
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse(("https", netloc, path, "", p.query, ""))


def html_to_text(html: str, url: str) -> str:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
    )
    return (text or "").strip()


def _strip_html_tags(text: str) -> str:
    """Plain text for chunking; README/markdown may still contain raw HTML tags."""
    if "<" not in text:
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator="\n", strip=True)


def _store_chunked_text(
    cur,
    url: str,
    title: str | None,
    text: str,
    kind: str,
) -> int:
    """Delete prior rows for url, insert chunks. Returns chunk count, or 0 if nothing stored."""
    text = _strip_html_tags(text).strip()
    chunks = chunk_text(text)
    if not chunks:
        return 0
    cur.execute(DELETE_DOCUMENTS_FOR_SOURCE_URL, (url,))
    n_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        meta = {
            "kind": kind,
            "chunk_index": i,
            "num_chunks": n_chunks,
        }
        cur.execute(
            INSERT_DOCUMENT_CHUNK,
            (url, i, title, chunk, Json(meta)),
        )
    return n_chunks


def fetch(client: httpx.Client, url: str) -> tuple[str | None, str | None]:
    try:
        r = client.get(url, follow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        if "text/html" not in ctype and "application/xhtml" not in ctype:
            return None, f"skip non-html content-type: {ctype!r}"
        return r.text, None
    except httpx.HTTPError as e:
        return None, str(e)


def main() -> None:
    pages_stored = 0
    client = httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_S,
    )
    conn = get_connection()
    cur = conn.cursor()

    try:
        print("GitHub: README + one hop of dev links from README (API + raw)…")
        try:
            for url, title, text in iter_github_readme_docs(
                client,
                GITHUB_OWNER,
                GITHUB_REPO,
                USER_AGENT,
                timeout=REQUEST_TIMEOUT_S,
            ):
                if pages_stored >= MAX_PAGES:
                    break
                time.sleep(DELAY_S)
                n = _store_chunked_text(cur, url, title, text, "github_readme")
                if not n:
                    print(f"[github skip] {url}")
                    continue
                conn.commit()
                pages_stored += 1
                print(f"[github ok {pages_stored}] {url} ({n} chunks)")
        except httpx.HTTPError as e:
            print(f"[github error] {e!r}")

        print("Web seeds (explicit URLs only, no link following)…")
        seen_seeds: set[str] = set()
        for s in SEED_URLS:
            if pages_stored >= MAX_PAGES:
                break
            url = canonical_url(s)
            if not url or url in seen_seeds:
                continue
            seen_seeds.add(url)

            time.sleep(DELAY_S)
            html, err = fetch(client, url)
            if html is None:
                print(f"[skip] {url} — {err}")
                continue

            text = html_to_text(html, url)
            if not text:
                print(f"[empty text] {url}")
                continue

            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else None

            n_chunks = _store_chunked_text(cur, url, title, text, "html")
            if not n_chunks:
                print(f"[no chunks] {url}")
                continue
            conn.commit()
            pages_stored += 1
            print(f"[ok {pages_stored}] {url} ({n_chunks} chunks)")
    finally:
        cur.close()
        conn.close()
        client.close()

    print(f"Done. Sources stored (chunked): {pages_stored}")
    print(
        "Next step: compute embeddings — from repo root: "
        "cd corpus && python3.13.12 embed.py (or `.venv/bin/python` from a 3.13.12 venv)."
    )


if __name__ == "__main__":
    main()
