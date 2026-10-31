import re
from difflib import SequenceMatcher

from app.database import SessionLocal
from app.models import Paper, ProjectPaper
from app.rag.ingestion import ingest_pdf, is_paper_indexed
from app.services.arxiv import normalize_arxiv_id, search_papers
from app.services.papers import save_papers


TITLE_STOP_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in",
    "into", "is", "of", "on", "or", "the", "to", "via", "with",
}


def normalize_title(title: str) -> str:
    """Normalize punctuation and whitespace for title comparisons."""
    return " ".join(re.findall(r"[a-z0-9]+", (title or "").casefold()))


def _title_match_score(query: str, title: str) -> float:
    """Score title similarity without using abstract-level semantics."""
    normalized_query = normalize_title(query)
    normalized_title = normalize_title(title)

    if not normalized_query or not normalized_title:
        return 0.0
    if normalized_query == normalized_title:
        return 1.0
    if normalized_query in normalized_title or normalized_title in normalized_query:
        return 0.96

    query_tokens = {
        token
        for token in normalized_query.split()
        if token not in TITLE_STOP_WORDS
    }
    title_tokens = {
        token
        for token in normalized_title.split()
        if token not in TITLE_STOP_WORDS
    }

    if not query_tokens or not title_tokens:
        return SequenceMatcher(None, normalized_query, normalized_title).ratio()

    overlap = len(query_tokens & title_tokens)
    if not overlap:
        return 0.0

    precision = overlap / len(query_tokens)
    recall = overlap / len(title_tokens)
    token_f1 = 2 * precision * recall / (precision + recall)
    sequence_score = SequenceMatcher(
        None,
        normalized_query,
        normalized_title,
    ).ratio()

    # Permit a model to include a small amount of extra search context while
    # still requiring most of the candidate's distinctive title tokens.
    title_coverage_score = recall * 0.85 if len(query_tokens) >= 2 else 0.0
    return max(token_f1, sequence_score * 0.9, title_coverage_score)


def _arxiv_id_key(value: str | None) -> str | None:
    normalized_id = normalize_arxiv_id(value or "")
    if not normalized_id:
        return None
    return re.sub(r"v\d+$", "", normalized_id).casefold()


def resolve_paper_title(paper_title: str, papers: list):
    """
    Resolve a user/model-supplied paper title against papers
    saved in the current project.

    Returns:
        (paper, None) on a confident match
        (None, error_dict) when no confident match exists
    """

    paper_title = (paper_title or "").strip()

    if not paper_title:
        return None, {
            "error": "A paper title or arXiv ID is required.",
            "needs_clarification": True,
            "candidate_titles": [paper.title for paper in papers[:5]],
        }

    if not papers:
        return None, {
            "error": "This paper is not saved in the project yet.",
            "paper_not_found": True,
            "candidate_titles": [],
        }

    requested_id = _arxiv_id_key(paper_title)
    if requested_id:
        for paper in papers:
            if _arxiv_id_key(paper.arxiv_id) == requested_id:
                return paper, None

    # Prefer exact and strong lexical title matches. Semantic matching can
    # confuse papers that discuss or cite the same field-defining work.
    normalized_title = normalize_title(paper_title)
    for paper in papers:
        if normalize_title(paper.title) == normalized_title:
            return paper, None

    ranked = sorted(
        ((_title_match_score(paper_title, paper.title), paper) for paper in papers),
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best_paper = ranked[0]
    if best_score < 0.72:
        return None, {
            "error": "This paper is not saved in the project yet.",
            "paper_not_found": True,
            "candidate_titles": [
                paper.title for score, paper in ranked[:3] if score >= 0.35
            ],
        }

    if len(ranked) > 1 and best_score - ranked[1][0] < 0.08:
        return None, {
            "error": "Multiple saved papers may match this description.",
            "needs_clarification": True,
            "candidate_titles": [
                paper.title for _, paper in ranked[:3]
            ],
        }

    return best_paper, None


def load_saved_papers(project_id: int) -> list:
    db = SessionLocal()

    try:
        return (
            db.query(Paper)
            .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
            .filter(ProjectPaper.project_id == project_id)
            .all()
        )
    finally:
        db.close()


def ensure_paper_ready_for_rag(paper, project_id: int) -> dict | None:
    """
    Make sure a saved paper can be used for retrieval.

    Returns an error dict when the paper cannot be indexed,
    or None when chunks are available.
    """

    if not paper.pdf_url:
        return {
            "error": "This paper does not have a PDF URL available.",
            "paper_title": paper.title,
            "grounded": False,
        }

    if is_paper_indexed(paper.id, project_id):
        return None

    try:
        ingestion_result = ingest_pdf(
            pdf_url=paper.pdf_url,
            paper_id=paper.id,
            project_id=project_id,
            arxiv_id=paper.arxiv_id,
        )

        if not is_paper_indexed(paper.id, project_id):
            return {
                "error": (
                    "The paper could not be indexed successfully. "
                    "No evidence is available for this request."
                ),
                "paper_title": paper.title,
                "ingestion_result": ingestion_result,
                "grounded": False,
            }

    except Exception as exc:
        return {
            "error": "Failed to ingest the paper PDF.",
            "paper_title": paper.title,
            "details": str(exc),
            "grounded": False,
        }

    return None


def search_and_ingest_requested_paper(
    paper_title: str,
    project_id: int,
):
    """Find one explicitly requested paper, persist it, and index its PDF."""
    query = (paper_title or "").strip()
    if not query:
        return None, {
            "error": "A paper title or arXiv ID is required.",
            "needs_clarification": True,
            "candidate_titles": [],
            "grounded": False,
        }

    id_match = re.search(
        r"(?i)(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)",
        query,
    )
    if id_match:
        query = id_match.group(1)

    # A content tool is resolving one requested work, not discovering
    # related literature. Keep the lookup in the title field so a paper
    # that merely cites or discusses the requested work cannot be selected.
    candidates = search_papers(query=query, limit=10, search_mode="title")
    if not candidates:
        return None, {
            "error": "No matching paper was found on arXiv.",
            "needs_clarification": True,
            "candidate_titles": [],
            "grounded": False,
        }

    requested_id = _arxiv_id_key(query)
    if requested_id:
        ranked = [
            (1.0 if _arxiv_id_key(paper.get("arxiv_id")) == requested_id else 0.0, paper)
            for paper in candidates
        ]
    else:
        ranked = [
            (_title_match_score(query, paper.get("title") or ""), paper)
            for paper in candidates
        ]
    ranked.sort(key=lambda item: item[0], reverse=True)

    exact_matches = [
        paper
        for score, paper in ranked
        if score == 1.0
    ]
    if exact_matches:
        selected = exact_matches[0]
    else:
        best_score, selected = ranked[0]
        if best_score < 0.72:
            return None, {
                "error": (
                    "I could not confidently identify the requested paper. "
                    "Please provide its exact title or arXiv ID."
                ),
                "needs_clarification": True,
                "candidate_titles": [
                    paper.get("title")
                    for score, paper in ranked[:5]
                    if score >= 0.35 and paper.get("title")
                ],
                "grounded": False,
            }

        if len(ranked) > 1 and best_score - ranked[1][0] < 0.08:
            return None, {
                "error": "Several arXiv papers may match that title.",
                "needs_clarification": True,
                "candidate_titles": [
                    paper.get("title")
                    for score, paper in ranked[:5]
                    if score >= best_score - 0.08 and paper.get("title")
                ],
                "grounded": False,
            }

    save_papers([selected], project_id)

    # Re-read the canonical record from the project database. The content
    # tool receives only this persisted paper ID after indexing succeeds.
    selected_id = _arxiv_id_key(selected.get("arxiv_id"))
    saved_paper = next(
        (
            paper
            for paper in load_saved_papers(project_id)
            if _arxiv_id_key(paper.arxiv_id) == selected_id
        ),
        None,
    )
    if saved_paper is None:
        return None, {
            "error": "The paper was found but could not be loaded from the project database.",
            "grounded": False,
        }

    indexing_error = ensure_paper_ready_for_rag(saved_paper, project_id)
    if indexing_error:
        return None, indexing_error

    return saved_paper, None


async def resolve_saved_paper(paper_title: str, project_id: int):
    papers = load_saved_papers(project_id)
    paper, resolution_error = resolve_paper_title(
        paper_title=paper_title,
        papers=papers,
    )

    if resolution_error and resolution_error.get("needs_clarification"):
        return None, resolution_error

    if paper is None and resolution_error and resolution_error.get("paper_not_found"):
        paper, resolution_error = search_and_ingest_requested_paper(
            paper_title=paper_title,
            project_id=project_id,
        )

    if resolution_error:
        return None, resolution_error

    indexing_error = ensure_paper_ready_for_rag(paper, project_id)

    if indexing_error:
        return None, indexing_error

    return paper, None


async def resolve_saved_papers(paper_titles: list[str], project_id: int):
    if not isinstance(paper_titles, list) or len(paper_titles) < 2:
        return None, {
            "error": "At least two paper titles are required.",
            "needs_clarification": True,
            "candidate_titles": [],
            "grounded": False,
        }

    normalized_titles = [normalize_title(title) for title in paper_titles]
    if len(normalized_titles) != len(set(normalized_titles)):
        return None, {
            "error": "Duplicate paper titles were provided.",
            "grounded": False,
        }

    resolved = []
    resolved_ids = set()

    for paper_title in paper_titles:
        paper, error = await resolve_saved_paper(
            paper_title=paper_title,
            project_id=project_id,
        )

        if error:
            return None, {
                **error,
                "requested_title": paper_title,
            }

        if paper.id in resolved_ids:
            return None, {
                "error": "The same saved paper was selected more than once.",
                "requested_title": paper_title,
                "grounded": False,
            }

        resolved.append(paper)
        resolved_ids.add(paper.id)

    return resolved, None


def paper_metadata(paper) -> dict:
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "published": (
            paper.published.isoformat()
            if paper.published
            else None
        ),
        "paper_url": paper.paper_url,
        "pdf_url": paper.pdf_url,
    }


def paper_identity(paper) -> dict:
    return {
        "paper_id": paper.id,
        "paper_title": paper.title,
        **paper_metadata(paper),
    }
