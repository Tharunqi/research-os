from app.tools.evidence import (
    collect_sources,
    gather_multi_paper_evidence,
)
from app.tools.generate import generate_from_excerpts


def synthesize_literature(
    paper_ids: list[int],
    project_id: int,
    topic: str,
    top_k: int = 8,
) -> dict:
    """
    Synthesize findings across multiple papers on a topic.

    The synthesis is generated on demand and is not stored
    in the database.
    """

    topic_text = (topic or "").strip()

    if not topic_text:
        return {
            "synthesis": None,
            "sources": [],
            "grounded": False,
            "error": "A topic is required for literature synthesis.",
        }

    if len(paper_ids) < 2:
        return {
            "synthesis": None,
            "sources": [],
            "grounded": False,
            "error": "At least two papers are required for synthesis.",
        }

    if len(paper_ids) != len(set(paper_ids)):
        return {
            "synthesis": None,
            "sources": [],
            "grounded": False,
            "error": "Duplicate paper IDs were provided.",
        }

    evidence = gather_multi_paper_evidence(
        paper_ids=paper_ids,
        project_id=project_id,
        question=(
            f"Find claims, methods, results, and open questions related "
            f"to this topic: {topic_text}"
        ),
        top_k=top_k,
    )

    if evidence["error"]:
        return {
            "synthesis": None,
            "sources": collect_sources(evidence["chunks"]),
            "grounded": False,
            "error": evidence["error"],
        }

    synthesis = generate_from_excerpts(
        instructions=(
            "You are an academic research assistant synthesizing "
            "literature. Use only the supplied excerpts. Attribute "
            "every claim to the paper it came from. Never transfer a "
            "finding from one paper to another. Distinguish agreement, "
            "disagreement, and missing evidence. Do not invent papers, "
            "citations, or results. Organize the synthesis under these "
            "headings:\n"
            "1. Shared Problem Framing\n"
            "2. Points of Agreement\n"
            "3. Points of Disagreement or Divergence\n"
            "4. Complementary Methods or Findings\n"
            "5. Open Questions and Gaps\n"
            "6. Concise Synthesis\n\n"
            "Cite claims using [Paper ID: X, Page: Y], exactly matching "
            "the identifiers and page numbers in the supplied excerpts."
        ),
        user_input=(
            f"Synthesize the literature on this topic: {topic_text}\n\n"
            "Use only the provided excerpts.\n\n"
            + "\n\n".join(evidence["contexts"])
        ),
    )

    return {
        "synthesis": synthesis,
        "topic": topic_text,
        "sources": collect_sources(evidence["chunks"]),
        "grounded": True,
    }
