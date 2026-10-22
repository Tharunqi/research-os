
from sqlalchemy import text
from app.database import engine
import os
import tempfile
from pathlib import Path

import requests
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

from app.rag.vector_store import get_embed_model, get_vector_store


MAX_PDF_SIZE = 50 * 1024 * 1024  # 50 MB
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def download_pdf(pdf_url: str) -> Path:
    """Download a PDF to a temporary file and return its path."""
    response = requests.get(
        pdf_url,
        timeout=(10, 60),
        stream=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
        response.close()
        raise ValueError("The URL does not appear to point to a PDF.")

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False,
    )
    path = Path(temp_file.name)
    total_bytes = 0

    try:
        with temp_file:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue

                total_bytes += len(chunk)
                if total_bytes > MAX_PDF_SIZE:
                    raise ValueError("PDF exceeds the 50 MB size limit.")

                temp_file.write(chunk)

        if total_bytes == 0:
            raise ValueError("The downloaded PDF is empty.")

        return path

    except Exception:
        path.unlink(missing_ok=True)
        raise

    finally:
        response.close()


def is_paper_indexed(paper_id: int, project_id: int) -> bool:
    """
    Check whether a paper already has indexed chunks
    for the given project.
    """
    with engine.connect() as connection:
        # The vector table may not exist before the first ingestion.
        table_exists = connection.execute(
            text("SELECT to_regclass('public.data_researchos_chunks')")
        ).scalar_one()

        if table_exists is None:
            return False

        chunk_count = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM public.data_researchos_chunks
                WHERE metadata_->>'paper_id' = :paper_id
                  AND metadata_->>'project_id' = :project_id
            """),
            {
                "paper_id": str(paper_id),
                "project_id": str(project_id),
            },
        ).scalar_one()

        return chunk_count > 0



def ingest_pdf(
    pdf_url: str,
    paper_id: int,
    project_id: int,
    arxiv_id: str,
) -> dict:
    """
    Extract a paper, split it into chunks, embed the chunks,
    and store them in PostgreSQL using LlamaIndex.
    """
    if paper_id <= 0 or project_id <= 0:
        raise ValueError("paper_id and project_id must be positive integers.")

    pdf_path = download_pdf(pdf_url)

    try:
        # Step 1: Extract one document per PDF page.
        documents = PDFReader().load_data(file=pdf_path)

        if not documents:
            raise ValueError("No readable pages were found in the PDF.")

        # Step 2: Attach identifiers used to filter retrieval later.
        for page_index, document in enumerate(documents, start=1):
            document.metadata.update(
                {
                    "paper_id": paper_id,
                    "project_id": project_id,
                    "arxiv_id": arxiv_id,
                    "page_number": page_index,
                }
            )

        # Step 3: Split page text into overlapping chunks.
        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # Step 4: Configure embeddings and PostgreSQL storage.
        vector_store = get_vector_store()
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # Step 5: Embed and persist the chunks.
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[splitter],
            embed_model=get_embed_model(),
            show_progress=True,
        )

        # Count the chunks produced by the splitter.
        chunks = splitter.get_nodes_from_documents(documents)

        return {
            "paper_id": paper_id,
            "project_id": project_id,
            "arxiv_id": arxiv_id,
            "pages_processed": len(documents),
            "chunks_created": len(chunks),
            "status": "indexed",
        }

    finally:
        pdf_path.unlink(missing_ok=True)
