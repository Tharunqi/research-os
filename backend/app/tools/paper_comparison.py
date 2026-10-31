from app.tools.evidence import (
    collect_sources,
    gather_multi_paper_evidence,
)
from app.tools.generate import generate_from_excerpts


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

    evidence = gather_multi_paper_evidence(
        paper_ids=paper_ids,
        project_id=project_id,
        question=(
            "Find the paper's research problem, proposed method, "
            "main contributions, datasets, experimental setup, "
            "results, baselines, and limitations."
        ),
        top_k=top_k,
    )

    if evidence["error"]:
        return {
            "comparison": None,
            "sources": collect_sources(evidence["chunks"]),
            "grounded": False,
            "error": evidence["error"],
        }

    comparison = generate_from_excerpts(
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
        user_input=(
            "Compare the following academic papers using only the "
            "provided excerpts.\n\n"
            + "\n\n".join(evidence["contexts"])
        ),
    )

    return {
        "comparison": comparison,
        "sources": collect_sources(evidence["chunks"]),
        "grounded": True,
    }
