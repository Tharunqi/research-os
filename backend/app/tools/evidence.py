from app.rag.retrieval import retrieve_chunks


def collect_sources(chunks: list[dict]) -> list[dict]:
    """Deduplicate retrieved chunks into page-level source records."""

    sources = []
    seen = set()

    for chunk in chunks:
        key = (chunk["paper_id"], chunk["page_number"])

        if key in seen:
            continue

        seen.add(key)
        sources.append(
            {
                "paper_id": chunk["paper_id"],
                "page_number": chunk["page_number"],
                "arxiv_id": chunk["arxiv_id"],
            }
        )

    return sources


def retrieve_paper_chunks(
    paper_id: int,
    project_id: int,
    question: str,
    top_k: int,
) -> list[dict]:
    """Retrieve evidence chunks for one paper in one project."""

    return retrieve_chunks(
        question=question,
        paper_id=paper_id,
        project_id=project_id,
        top_k=top_k,
    )


def format_single_paper_excerpts(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Page {chunk['page_number']}]\n{chunk['text']}"
        for chunk in chunks
    )


def format_labeled_paper_excerpts(
    paper_id: int,
    chunks: list[dict],
) -> str:
    excerpts = []

    for chunk in chunks:
        excerpts.append(
            f"[Paper ID: {paper_id}, Page: {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )

    return (
        f"=== PAPER ID: {paper_id} ===\n"
        + "\n\n".join(excerpts)
    )


def gather_multi_paper_evidence(
    paper_ids: list[int],
    project_id: int,
    question: str,
    top_k: int,
) -> dict:
    """
    Retrieve evidence for each paper separately.

    Returns an error if any paper has no retrieved chunks so claims
    cannot be attributed incorrectly.
    """

    paper_contexts = []
    all_chunks = []

    for paper_id in paper_ids:
        chunks = retrieve_paper_chunks(
            paper_id=paper_id,
            project_id=project_id,
            question=question,
            top_k=top_k,
        )

        if not chunks:
            return {
                "contexts": [],
                "chunks": all_chunks,
                "error": (
                    f"No relevant content was found for paper ID "
                    f"{paper_id}. Ensure the paper has been indexed."
                ),
            }

        all_chunks.extend(chunks)
        paper_contexts.append(
            format_labeled_paper_excerpts(paper_id, chunks)
        )

    return {
        "contexts": paper_contexts,
        "chunks": all_chunks,
        "error": None,
    }
