from app.rag.answer import answer_question
from app.tools.paper_resolution import paper_identity, resolve_saved_paper


async def answer_paper_question(
    question: str,
    paper_title: str,
    project_id: int
) -> dict:
    paper, error = await resolve_saved_paper(
        paper_title=paper_title,
        project_id=project_id,
    )

    if error:
        return error

    result = answer_question(
        question=question,
        paper_id=paper.id,
        project_id=project_id,
    )

    return {
        **paper_identity(paper),
        **result,
    }
