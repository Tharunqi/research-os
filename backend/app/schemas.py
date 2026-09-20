from pydantic import BaseModel, ConfigDict


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None


class PaperResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    title: str
    abstract: str | None
    authors: list[str]
    published: str | None
    paper_url: str | None
    pdf_url: str | None


class ResearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    papers: list[PaperResult]