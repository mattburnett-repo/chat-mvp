"""GitHub repo identity and explicit web page URLs to ingest (no link crawling)."""

GITHUB_OWNER = "civictechdc"
GITHUB_REPO = "cib-mango-tree"

# Each URL is fetched once as HTML, text-extracted, chunked, and stored.
# Add or remove entries here; no automatic link following.
SEED_URLS = [
    "https://cibmangotree.org/",
    "https://civictechdc.github.io/cib-mango-tree/",
    "https://civictechdc.github.io/cib-mango-tree/guides/contributing/new_contributor_guide/",
    "https://civictechdc.github.io/cib-mango-tree/guides/get-started/installation/",
    "https://civictechdc.github.io/cib-mango-tree/guides/contributing/ai_assisted_dev/",
]
