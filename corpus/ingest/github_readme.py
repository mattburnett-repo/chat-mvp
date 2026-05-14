"""
Ingest the GitHub repo README plus one hop of developer-oriented files linked from it.

Uses the public GitHub API (rate-limited); no token required for public repos.
Does not crawl the GitHub website — only API + raw.githubusercontent.com.
"""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import unquote, urlparse

import httpx

GITHUB_BLOB_RE = re.compile(
    r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/([^?#]+)",
    re.I,
)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def _api_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _dev_file_path_allowed(repo_path: str) -> bool:
    """Keep docs/text/config useful for developers; drop images, fonts, etc."""
    lower = repo_path.lower()
    base = lower.rsplit("/", 1)[-1]
    if any(
        lower.endswith(ext)
        for ext in (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            ".webp",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".pdf",
            ".zip",
        )
    ):
        return False
    if any(repo_path.endswith(ext) for ext in (".md", ".rst", ".txt", ".adoc")):
        return True
    if base in ("license", "copying", "contributing", "code_of_conduct", "security"):
        return True
    if base.startswith("readme."):
        return True
    if lower.startswith("docs/"):
        return True
    if ".github/" in lower and lower.endswith((".md", ".yml", ".yaml")):
        return True
    return False


def _markdown_hrefs(markdown: str) -> list[str]:
    return [m[1].strip() for m in MARKDOWN_LINK_RE.findall(markdown)]


def _resolve_readme_href(
    href: str,
    owner: str,
    repo: str,
) -> str | None:
    """Map a README link to a repo-relative file path, or None to skip."""
    href = href.strip().split("#", 1)[0].strip()
    if not href or href.startswith("mailto:"):
        return None

    parsed = urlparse(href)
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    if not netloc:
        return unquote(href.lstrip("./")) or None

    if netloc == "github.com" or netloc.endswith(".github.com"):
        m = GITHUB_BLOB_RE.match(href)
        if not m:
            return None
        o, r, _ref, path = m.groups()
        if o.lower() != owner.lower() or r.lower() != repo.lower():
            return None
        return unquote(path)

    if netloc == "raw.githubusercontent.com":
        # /owner/repo/ref/path
        parts = (parsed.path or "").strip("/").split("/")
        if len(parts) < 4:
            return None
        o, r = parts[0], parts[1]
        if o.lower() != owner.lower() or r.lower() != repo.lower():
            return None
        return unquote("/".join(parts[3:]))

    return None


def _blob_page_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"


def iter_github_readme_docs(
    client: httpx.Client,
    owner: str,
    repo: str,
    user_agent: str,
    *,
    timeout: float,
) -> Iterator[tuple[str, str | None, str]]:
    """
    Yield (canonical_html_url, short_title, plain_text) for README and linked dev files.

    One hop from README only (no recursive walk of linked files).
    """
    h = _api_headers(user_agent)
    repo_url = f"https://api.github.com/repos/{owner}/{repo}"
    r_repo = client.get(repo_url, headers=h, timeout=timeout)
    r_repo.raise_for_status()
    default_branch = r_repo.json()["default_branch"]

    r_rm = client.get(f"{repo_url}/readme", headers=h, timeout=timeout)
    r_rm.raise_for_status()
    rm = r_rm.json()
    readme_path = rm["path"]
    readme_html_url = rm["html_url"]
    download_url = rm["download_url"]

    r_text = client.get(
        download_url,
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )
    r_text.raise_for_status()
    readme_body = r_text.text
    yield readme_html_url, rm.get("name"), readme_body

    seen: set[str] = {readme_path.replace("\\", "/").lower()}
    for href in _markdown_hrefs(readme_body):
        path = _resolve_readme_href(href, owner, repo)
        if not path:
            continue
        norm = path.replace("\\", "/").lower()
        if norm in seen:
            continue
        if not _dev_file_path_allowed(path):
            continue
        seen.add(norm)

        raw = (
            f"https://raw.githubusercontent.com/{owner}/{repo}/"
            f"{default_branch}/{path}"
        )
        try:
            rf = client.get(
                raw,
                headers={"User-Agent": user_agent},
                timeout=timeout,
            )
            rf.raise_for_status()
        except httpx.HTTPError:
            continue
        if b"\x00" in rf.content[:4096]:
            continue
        try:
            text = rf.content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not text.strip():
            continue
        title = path.rsplit("/", 1)[-1]
        yield _blob_page_url(owner, repo, default_branch, path), title, text
