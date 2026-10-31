import re

from app.rag.ingestion import ingest_pdf, is_paper_indexed
from app.services.arxiv import normalize_arxiv_id, search_papers
from app.services.papers import save_papers
from app.tools.paper_resolution import (
    _title_match_score,
    normalize_title,
    paper_metadata,
)


def search_and_save_papers(
    project_id: int,
    query: str,
    limit: int = 5,
    search_mode: str = "topic",
):
    query = (query or "").strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if search_mode not in {"title", "topic"}:
        raise ValueError("search_mode must be either 'title' or 'topic'.")

    limit = max(1, min(int(limit), 10))

    # If the query is an arXiv ID or URL, prioritize the ID.
    normalized_query_id = normalize_arxiv_id(query)
    match = re.search(
        r"(?i)(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)",
        query,
    )

    if match:
        query = match.group(1)
    elif normalized_query_id:
        query = normalized_query_id
    identifier_query = bool(match or normalized_query_id)

    print("Final arXiv search query:", query)

    papers = search_papers(
        query=query,
        limit=limit,
        search_mode=search_mode,
    )

    # A title query must not silently become a broad full-text search.
    # That is how a paper that merely cites the requested work can be
    # mistaken for the requested paper (for example, Music Transformer
    # for Attention Is All You Need).
    if search_mode == "title" and not identifier_query and papers:
        ranked = sorted(
            (
                (_title_match_score(query, paper.get("title") or ""), paper)
                for paper in papers
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        exact_matches = [
            paper
            for _, paper in ranked
            if normalize_title(paper.get("title") or "")
            == normalize_title(query)
        ]
        if exact_matches:
            papers = exact_matches[:limit]
        else:
            papers = [paper for score, paper in ranked if score >= 0.72][:limit]

    saved_papers = save_papers(papers, project_id)

    results = []

    for paper in saved_papers:
        result = {
            **paper_metadata(paper),
            "ingestion_status": None,
            "ingestion_error": None,
        }

        if not paper.pdf_url:
            result["ingestion_status"] = "no_pdf_url"
        elif is_paper_indexed(paper.id, project_id):
            result["ingestion_status"] = "already_indexed"
        else:
            try:
                ingest_pdf(
                    pdf_url=paper.pdf_url,
                    paper_id=paper.id,
                    project_id=project_id,
                    arxiv_id=paper.arxiv_id,
                )

                if is_paper_indexed(paper.id, project_id):
                    result["ingestion_status"] = "indexed"
                else:
                    result["ingestion_status"] = "failed"
                    result["ingestion_error"] = (
                        "Ingestion finished but no chunks were stored."
                    )
            except Exception as exc:
                result["ingestion_status"] = "failed"
                result["ingestion_error"] = str(exc)

        results.append(result)

    return results
