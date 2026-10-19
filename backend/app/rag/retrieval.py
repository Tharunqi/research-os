
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from app.rag.vector_store import get_embed_model, get_vector_store


def retrieve_chunks(
    question: str,
    paper_id: int,
    project_id: int,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve relevant chunks from one paper within one project."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if paper_id <= 0 or project_id <= 0:
        raise ValueError("paper_id and project_id must be positive.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Reconnect to the existing vector index; don't re-embed documents.
    index = VectorStoreIndex.from_vector_store(
        vector_store=get_vector_store(),
        embed_model=get_embed_model(),
    )

    # Both filters are required to prevent cross-paper/project retrieval.
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="paper_id",
                value=paper_id,
                operator=FilterOperator.EQ,
            ),
            MetadataFilter(
                key="project_id",
                value=project_id,
                operator=FilterOperator.EQ,
            ),
        ],
        condition="and",
    )

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    )

    nodes = retriever.retrieve(question)

    return [
        {
            "text": item.node.get_content(),
            "paper_id": item.node.metadata.get("paper_id"),
            "project_id": item.node.metadata.get("project_id"),
            "arxiv_id": item.node.metadata.get("arxiv_id"),
            "page_number": item.node.metadata.get("page_number"),
            "score": item.score,
        }
        for item in nodes
    ]
