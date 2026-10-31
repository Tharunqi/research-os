
import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse

import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Supports modern IDs (1706.03762) and legacy IDs (hep-th/9901001),
# optionally including a version suffix (v2).
ARXIV_ID_PATTERN = re.compile(
    r"^(?P<base>(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7}))"
    r"(?P<version>v\d+)?$",
    re.IGNORECASE,
)


def normalize_arxiv_id(value: str) -> str | None:
    """Extract a valid arXiv identifier from an ID or arXiv URL."""
    value = unquote(value.strip())

    if value.lower().startswith("arxiv:"):
        value = value[6:].strip()

    if "://" in value:
        parsed = urlparse(value)
        if not parsed.hostname or not parsed.hostname.lower().endswith("arxiv.org"):
            return None

        value = parsed.path.strip("/")
        if value.startswith(("abs/", "pdf/")):
            value = value.split("/", 1)[1]

    value = value.removesuffix(".pdf").strip("/")
    match = ARXIV_ID_PATTERN.fullmatch(value)

    if not match:
        return None

    return match.group("base") + (match.group("version") or "")


def _parse_entry(entry: ET.Element) -> dict:
    """Convert one arXiv Atom entry into our existing paper format."""
    def text_of(tag: str) -> str | None:
        element = entry.find(f"atom:{tag}", ATOM_NS)
        return (
            " ".join(element.text.split())
            if element is not None and element.text
            else None
        )

    paper_url = None
    pdf_url = None

    for link in entry.findall("atom:link", ATOM_NS):
        href = link.get("href")
        if not href:
            continue

        if link.get("rel") == "alternate":
            paper_url = href
        elif link.get("type") == "application/pdf":
            pdf_url = href

    authors = [
        name.text.strip()
        for author in entry.findall("atom:author", ATOM_NS)
        if (name := author.find("atom:name", ATOM_NS)) is not None
        and name.text
    ]

    categories = [
        category.get("term")
        for category in entry.findall("atom:category", ATOM_NS)
        if category.get("term")
    ]

    return {
        "arxiv_id": text_of("id"),
        "title": text_of("title"),
        "abstract": text_of("summary"),
        "published": text_of("published"),
        "authors": authors,
        "paper_url": paper_url,
        "pdf_url": pdf_url,
        "categories": categories,
    }


def _query_api(
    *,
    search_query: str | None = None,
    id_list: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """Execute one arXiv API query and parse its results."""
    params = {"start": 0, "max_results": limit}

    if id_list:
        params["id_list"] = id_list
    elif search_query:
        params["search_query"] = search_query
    else:
        raise ValueError("Provide either an arXiv ID or a search query.")

    response = requests.get(
        ARXIV_API_URL,
        params=params,
        timeout=(5, 30),
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)

    return [
        _parse_entry(entry)
        for entry in root.findall("atom:entry", ATOM_NS)
    ]


def _title_key(title: str) -> str:
    """Normalize titles for exact-title comparison."""
    return " ".join(re.findall(r"\w+", title.casefold()))


def search_papers(
    query: str,
    limit: int = 5,
    search_mode: str = "topic",
) -> list[dict]:
    """
    Search arXiv by exact identifier or title-first text search.

    Exact IDs are resolved through id_list, not full-text search.
    ``title`` searches stay in the title field. ``topic`` searches fall
    back to title/abstract matches when the title search has no exact hit.
    """
    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")
    if search_mode not in {"title", "topic"}:
        raise ValueError("search_mode must be either 'title' or 'topic'.")

    # Case 1: The input is an arXiv ID or arXiv URL.
    arxiv_id = normalize_arxiv_id(query)

    if arxiv_id:
        results = _query_api(id_list=arxiv_id, limit=1)

        # If the user supplied a version, require that exact version.
# If they supplied a base ID, accept the version returned by arXiv.
        requested_has_version = bool(
            re.search(r"v\d+$", arxiv_id)
        )

        exact_results = []

        for paper in results:
            returned_id = normalize_arxiv_id(paper["arxiv_id"] or "")

            if not returned_id:
                continue

            if requested_has_version:
                matches = returned_id == arxiv_id
            else:
                returned_base_id = re.sub(r"v\d+$", "", returned_id)
                matches = returned_base_id == arxiv_id

            if matches:
                exact_results.append(paper)

        return exact_results

    # Case 2: Search the title field before broad full-text search.
    escaped_query = query.replace("\\", "\\\\").replace('"', '\\"')
    title_results = _query_api(
        search_query=f'ti:"{escaped_query}"',
        limit=limit,
    )

    requested_title = _title_key(query)

    exact_title_results = [
        paper
        for paper in title_results
        if paper["title"]
        and _title_key(paper["title"]) == requested_title
    ]

    # If the title matches exactly, do not mix in related papers.
    if exact_title_results:
        return exact_title_results

    # Resolving a specific requested paper must never turn into a broad
    # full-text search. That fallback can return papers which only cite the
    # requested work, and adds a second network request before ingestion.
    if search_mode == "title":
        return title_results

    # Case 3: No exact title match: fall back to broader search.
    broad_results = _query_api(
        search_query=f"all:{query}",
        limit=limit,
    )

    # Merge candidates without returning duplicate arXiv IDs.
    combined = []
    seen_ids = set()

    for paper in title_results + broad_results:
        paper_id = normalize_arxiv_id(paper["arxiv_id"] or "")
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            combined.append(paper)

    return combined[:limit]
