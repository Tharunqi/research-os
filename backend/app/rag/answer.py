
import os

from openai import OpenAI

from app.rag.retrieval import retrieve_chunks


def answer_question(
    question: str,
    paper_id: int,
    project_id: int,
    top_k: int = 5,
) -> dict:
    """Answer a question using retrieved evidence from one paper."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    chunks = retrieve_chunks(
        question=question,
        paper_id=paper_id,
        project_id=project_id,
        top_k=top_k,
    )

    if not chunks:
        return {
            "answer": "I couldn't find relevant evidence in this paper.",
            "sources": [],
            "grounded": False,
        }

    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i} | Page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    client = OpenAI()

    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    response = client.responses.create(
        model=model,
        instructions=(
            "You are ResearchOS, a research assistant. "
            "Answer using only the supplied paper excerpts. "
            "Do not use outside knowledge or invent facts. "
            "Cite supporting evidence using [Page N], where N is "
            "a page number present in the supplied excerpts. "
            "If the excerpts do not answer the question, say so "
            "clearly. Distinguish explicit findings from your "
            "interpretation. Format mathematical expressions with TeX "
            "delimiters: use $...$ inline and $$...$$ on separate lines "
            "for display equations. Do not leave raw TeX commands un-delimited."
        ),
        input=(
            f"Question:\n{question}\n\n"
            f"Paper excerpts:\n{context}"
        ),
    )

    answer = response.output_text.strip()

    sources_by_page = {}

    for chunk in chunks:
        page = chunk["page_number"]
        sources_by_page[page] = {
            "page_number": page,
            "arxiv_id": chunk["arxiv_id"],
            "paper_id": chunk["paper_id"],
        }

    return {
        "answer": answer,
        "sources": list(sources_by_page.values()),
        "grounded": True,
    }
