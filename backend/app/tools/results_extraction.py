from app.tools.evidence import (
    collect_sources,
    format_single_paper_excerpts,
    retrieve_paper_chunks,
)
from app.tools.generate import generate_from_excerpts


def extract_paper_results(
    paper_id: int,
    project_id: int,
    focus: str | None = None,
    top_k: int = 15,
) -> dict:
    """
    Extract reported experimental results from one paper.

    The extraction is generated on demand and is not stored
    in the database.
    """

    focus_text = (focus or "").strip()
    retrieval_question = (
        "Extract the paper's datasets, metrics, baselines, "
        "experimental results, quantitative findings, ablations, "
        "and reported limitations of the evaluation."
    )

    if focus_text:
        retrieval_question = (
            f"{retrieval_question} Pay special attention to: {focus_text}"
        )

    chunks = retrieve_paper_chunks(
        paper_id=paper_id,
        project_id=project_id,
        question=retrieval_question,
        top_k=top_k,
    )

    if not chunks:
        return {
            "results": None,
            "sources": [],
            "grounded": False,
            "error": "No relevant paper content was found.",
        }

    user_input = (
        "Extract the experimental results from this paper using "
        "only the following excerpts.\n\n"
        f"Paper excerpts:\n{format_single_paper_excerpts(chunks)}"
    )

    if focus_text:
        user_input = (
            f"Extraction focus requested by the user: {focus_text}\n\n"
            + user_input
        )

    results = generate_from_excerpts(
        instructions=(
            "You are a research assistant extracting experimental "
            "results from an academic paper. Use only the supplied "
            "excerpts. Do not invent numbers, datasets, or claims. "
            "If a value is not present, say it was not found. "
            "Copy numerical results exactly as written in the excerpts. "
            "Organize the extraction under these headings:\n"
            "1. Datasets and Benchmarks\n"
            "2. Metrics\n"
            "3. Baselines and Comparisons\n"
            "4. Main Quantitative Results\n"
            "5. Ablations or Additional Experiments\n"
            "6. Evaluation Limitations\n"
            "Cite supporting claims using [Page N], where N is a page "
            "number present in the supplied excerpts."
        ),
        user_input=user_input,
    )

    return {
        "results": results,
        "sources": collect_sources(chunks),
        "grounded": True,
        "focus": focus_text or None,
    }
