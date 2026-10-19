
from datetime import datetime

from app.database import SessionLocal
from app.models import Paper, Project, ProjectPaper


def save_papers(papers, project_id: int):
    db = SessionLocal()

    try:
        # Make sure the target project exists.
        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise ValueError(f"Project {project_id} does not exist")

        saved_papers = []
        seen_paper_ids = set()

        for paper_data in papers:
            arxiv_id = paper_data["arxiv_id"]

            # 1. Find the globally stored paper.
            paper = (
                db.query(Paper)
                .filter(Paper.arxiv_id == arxiv_id)
                .first()
            )

            # 2. Create it only if it does not exist.
            if paper is None:
                published = None

                if paper_data.get("published"):
                    published = datetime.fromisoformat(
                        paper_data["published"].replace(
                            "Z", "+00:00"
                        )
                    )

                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=paper_data["title"],
                    abstract=paper_data.get("abstract"),
                    authors=paper_data["authors"],
                    published=published,
                    paper_url=paper_data.get("paper_url"),
                    pdf_url=paper_data.get("pdf_url"),
                )

                db.add(paper)
                db.flush()  # Assign paper.id before linking it.

            # 3. Check whether this project-paper link exists.
            association = (
                db.query(ProjectPaper)
                .filter(
                    ProjectPaper.project_id == project_id,
                    ProjectPaper.paper_id == paper.id,
                )
                .first()
            )

            # 4. Create the association only if needed.
            if association is None:
                association = ProjectPaper(
                    project_id=project_id,
                    paper_id=paper.id,
                )
                db.add(association)

            if paper.id not in seen_paper_ids:
                saved_papers.append(paper)
                seen_paper_ids.add(paper.id)

        db.commit()

        for paper in saved_papers:
            db.refresh(paper)

        return saved_papers

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
