from app.tools.evidence import (
    collect_sources,
    format_single_paper_excerpts,
    retrieve_paper_chunks,
)
from app.tools.generate import generate_from_excerpts


def summarize_paper(
    paper_id: int,
    project_id: int,
    top_k: int = 15,
) -> dict:
    """
    Generate a structured summary of a paper using retrieved PDF chunks.
    The summary is generated on demand and is not stored in the database.
    """

    chunks = retrieve_paper_chunks(
        paper_id=paper_id,
        project_id=project_id,
        question=(
            "Summarize the paper's problem, motivation, methodology, "
            "main contributions, experiments, results, and limitations."
        ),
        top_k=top_k,
    )

    if not chunks:
        return {
            "summary": None,
            "sources": [],
            "grounded": False,
            "error": "No relevant paper content was found.",
        }

    summary = generate_from_excerpts(
        instructions=(
            "You are a research assistant summarizing an academic paper. "
            "Use only the supplied excerpts. Do not invent facts or use "
            "outside knowledge. If a requested detail is not available "
            "in the excerpts, explicitly say it was not found. "
            "Organize the summary under these headings:\n"
            "1. Problem and Motivation\n"
            "2. Proposed Method\n"
            "3. Key Contributions\n"
            "4. Experimental Setup\n"
            "5. Main Results\n"
            "6. Limitations\n"
            "Cite supporting claims using [Page N], where N is a page "
            "number present in the supplied excerpts."
        ),
        user_input=(
            "Summarize this academic paper using the following excerpts.\n\n"
            f"Paper excerpts:\n{format_single_paper_excerpts(chunks)}"
        ),
    )

    return {
        "summary": summary,
        "sources": collect_sources(chunks),
        "grounded": True,
    }
