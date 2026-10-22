
import os

from openai import OpenAI

from app.rag.retrieval import retrieve_chunks


def compare_papers(
    paper_ids: list[int],
    project_id: int,
    top_k: int = 8,
) -> dict:
    """
    Compare multiple papers using retrieved PDF excerpts.

    The comparison is generated on demand and is not stored
    in the database.
    """

    if len(paper_ids) < 2:
        return {
            "comparison": None,
            "sources": [],
            "grounded": False,
            "error": "At least two papers are required for comparison.",
        }

    if len(paper_ids) != len(set(paper_ids)):
        return {
            "comparison": None,
            "sources": [],
            "grounded": False,
            "error": "Duplicate paper IDs were provided.",
        }

    paper_contexts = []
    all_sources = []

    question = (
        "Find the paper's research problem, proposed method, "
        "main contributions, datasets, experimental setup, "
        "results, baselines, and limitations."
    )

    for paper_id in paper_ids:
        chunks = retrieve_chunks(
            question=question,
            paper_id=paper_id,
            project_id=project_id,
            top_k=top_k,
        )

        if not chunks:
            return {
                "comparison": None,
                "sources": all_sources,
                "grounded": False,
                "error": (
                    f"No relevant content was found for paper ID "
                    f"{paper_id}. Ensure the paper has been indexed."
                ),
            }

        excerpts = []

        for chunk in chunks:
            page = chunk["page_number"]
            excerpts.append(
                f"[Paper ID: {paper_id}, Page: {page}]\n"
                f"{chunk['text']}"
            )

            source = {
                "paper_id": paper_id,
                "page_number": page,
                "arxiv_id": chunk["arxiv_id"],
            }

            if source not in all_sources:
                all_sources.append(source)

        paper_contexts.append(
            f"=== PAPER ID: {paper_id} ===\n"
            + "\n\n".join(excerpts)
        )

    context = "\n\n".join(paper_contexts)

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    response = client.responses.create(
        model=model,
        instructions=(
            "You are an academic research assistant comparing papers. "
            "Use only the supplied excerpts. Never transfer a method, "
            "dataset, result, or claim from one paper to another. "
            "Distinguish reported facts from interpretations. "
            "If information is missing, explicitly state that it was "
            "not found in the supplied excerpts. Do not invent numerical "
            "results or claim that one method is better without evidence. "
            "Organize the comparison under these headings:\n"
            "1. Research Problem and Motivation\n"
            "2. Methodology and Key Differences\n"
            "3. Datasets and Experimental Setup\n"
            "4. Results and Baselines\n"
            "5. Strengths and Limitations\n"
            "6. Overall Comparison\n\n"
            "Cite claims using [Paper ID: X, Page: Y], exactly matching "
            "the identifiers and page numbers in the supplied excerpts."
        ),
        input=(
            "Compare the following academic papers using only the "
            "provided excerpts.\n\n"
            f"{context}"
        ),
    )

    return {
        "comparison": response.output_text.strip(),
        "sources": all_sources,
        "grounded": True,
    }
