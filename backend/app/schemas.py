from pydantic import BaseModel, ConfigDict
from datetime import datetime



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


class ResearchRequest(BaseModel):
    question: str
    project_id: int
    session_id: int
class ResearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    papers: list[PaperResult]

class ChatSessionCreate(BaseModel):
    title: str | None = None


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    title: str | None
    created_at: datetime


class ChatMessageCreate(BaseModel):
    content: str


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime