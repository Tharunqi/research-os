import requests
import xml.etree.ElementTree as ET


ARXIV_API_URL = "https://export.arxiv.org/api/query"


def search_papers(query: str, limit: int = 5):
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
    }

    response = requests.get(
        ARXIV_API_URL,
        params=params,
    )

    response.raise_for_status()

    root = ET.fromstring(response.text)

    namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    papers = []

    for entry in root.findall("atom:entry", namespace):
        title = entry.find("atom:title", namespace)
        summary = entry.find("atom:summary", namespace)
        published = entry.find("atom:published", namespace)
        paper_id = entry.find("atom:id", namespace)

        authors = []

        for author in entry.findall("atom:author", namespace):
            name = author.find("atom:name", namespace)

            if name is not None:
                authors.append(name.text)

        paper_url = None
        pdf_url = None

        for link in entry.findall("atom:link", namespace):
            rel = link.get("rel")
            link_type = link.get("type")
            href = link.get("href")

            if rel == "alternate":
                paper_url = href

            if link_type == "application/pdf":
                pdf_url = href

        categories = []

        for category in entry.findall("atom:category", namespace):
            term = category.get("term")

            if term:
                categories.append(term)

        papers.append({
            "arxiv_id": (
                paper_id.text
                if paper_id is not None
                else None
            ),
            "title": (
                title.text.strip()
                if title is not None
                else None
            ),
            "abstract": (
                summary.text.strip()
                if summary is not None
                else None
            ),
            "published": (
                published.text
                if published is not None
                else None
            ),
            "authors": authors,
            "paper_url": paper_url,
            "pdf_url": pdf_url,
            "categories": categories,
        })

    return papers