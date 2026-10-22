
import os

from openai import OpenAI

from app.rag.retrieval import retrieve_chunks


def summarize_paper(
    paper_id: int,
    project_id: int,
    top_k: int = 15,
) -> dict:
    """
    Generate a structured summary of a paper using retrieved PDF chunks.
    The summary is generated on demand and is not stored in the database.
    """

    chunks = retrieve_chunks(
        question=(
            "Summarize the paper's problem, motivation, methodology, "
            "main contributions, experiments, results, and limitations."
        ),
        paper_id=paper_id,
        project_id=project_id,
        top_k=top_k,
    )

    if not chunks:
        return {
            "summary": None,
            "sources": [],
            "grounded": False,
            "error": "No relevant paper content was found.",
        }

    context_parts = []

    for chunk in chunks:
        context_parts.append(
            f"[Page {chunk['page_number']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    response = client.responses.create(
        model=model,
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
        input=(
            "Summarize this academic paper using the following excerpts.\n\n"
            f"Paper excerpts:\n{context}"
        ),
    )

    sources_by_page = {}

    for chunk in chunks:
        page = chunk["page_number"]
        sources_by_page[page] = {
            "page_number": page,
            "paper_id": chunk["paper_id"],
            "arxiv_id": chunk["arxiv_id"],
        }

    return {
        "summary": response.output_text.strip(),
        "sources": list(sources_by_page.values()),
        "grounded": True,
    }
