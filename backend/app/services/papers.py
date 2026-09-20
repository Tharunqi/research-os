from datetime import datetime

from app.database import SessionLocal
from app.models import Paper


def save_papers(papers,project_id):
    db = SessionLocal()

    try:
        saved_papers = []

        for paper_data in papers:
            arxiv_id = paper_data["arxiv_id"]

            existing_paper = (
                db.query(Paper)
                .filter(Paper.arxiv_id == arxiv_id)
                .first()
            )

            if existing_paper:
                saved_papers.append(existing_paper)
                continue

            published = None

            if paper_data.get("published"):
                published = datetime.fromisoformat(
                    paper_data["published"].replace("Z", "+00:00")
                )

            paper = Paper(
                arxiv_id=arxiv_id,
                title=paper_data["title"],
                abstract=paper_data["abstract"],
                authors=paper_data["authors"],
                published=published,
                paper_url=paper_data["paper_url"],
                pdf_url=paper_data["pdf_url"],
                project_id=project_id
            )

            db.add(paper)
            saved_papers.append(paper)

        db.commit()

        for paper in saved_papers:
            db.refresh(paper)

        return saved_papers

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()