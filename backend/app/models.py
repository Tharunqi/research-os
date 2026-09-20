from datetime import datetime

from sqlalchemy import (
    String,
    Text,
    DateTime,
    JSON,
    ForeignKey,
    func,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now()
    )

    papers: Mapped[list["Paper"]] = relationship(
        back_populates="project"
    )

    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="project"
    )


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False
    )

    arxiv_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    abstract: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    authors: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False
    )

    published: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    paper_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    pdf_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now()
    )

    project: Mapped["Project"] = relationship(
        back_populates="papers"
    )
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False
    )

    title: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now()
    )

    project: Mapped["Project"] = relationship(
        back_populates="chat_sessions"
    )