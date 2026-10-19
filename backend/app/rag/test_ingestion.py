
import os

from dotenv import load_dotenv

from app.rag.ingestion import ingest_pdf

load_dotenv()


def main():
    # Replace these with a real paper already saved in your database.
    paper_id = int(os.environ["TEST_PAPER_ID"])
    project_id = int(os.environ["TEST_PROJECT_ID"])
    arxiv_id = os.environ["TEST_ARXIV_ID"]
    pdf_url = os.environ["TEST_PDF_URL"]

    result = ingest_pdf(
        pdf_url=pdf_url,
        paper_id=paper_id,
        project_id=project_id,
        arxiv_id=arxiv_id,
    )

    print(result)


if __name__ == "__main__":
    main()
