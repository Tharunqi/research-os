import inspect
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from app.database import SessionLocal
from app.models import ChatMessage, ChatSession, Paper, Project, ProjectPaper
from app.schemas import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ProjectCreate,
    ResearchRequest,
    ResearchResponse,
)
from app.services.context_builder import build_context
from app.tools import (
    answer_paper_question,
    compare_papers,
    extract_paper_results,
    search_and_save_papers,
    summarize_paper,
    synthesize_literature,
)
from app.tools.paper_resolution import (
    normalize_title,
    paper_identity,
    resolve_saved_paper,
    resolve_saved_papers,
)

load_dotenv()


# --------------------------------------------------
# 1. FastAPI application
# --------------------------------------------------

app = FastAPI()

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# --------------------------------------------------
# 2. OpenAI client
# --------------------------------------------------

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# 4. Tool result helpers and handlers
# --------------------------------------------------

def has_grounded_success(tool_results) -> bool:
    return any(
        isinstance(result, dict) and result.get("grounded") is True
        for result in tool_results
    )


def has_blocking_tool_result(tool_results) -> bool:
    """
    True when the model must not invent a paper-specific answer.

    Earlier skipped/failed calls are ignored if a later tool
    returned grounded evidence.
    """

    if has_grounded_success(tool_results):
        return False

    for result in tool_results:
        if not isinstance(result, dict):
            continue

        if result.get("skipped") is True:
            continue

        if result.get("grounded") is False:
            return True

        if result.get("needs_clarification") is True:
            return True

        if "error" in result:
            return True

    return False


async def summarize_saved_paper(
    paper_title: str,
    project_id: int,
) -> dict:
    paper, error = await resolve_saved_paper(
        paper_title=paper_title,
        project_id=project_id,
    )

    if error:
        return error

    result = summarize_paper(
        paper_id=paper.id,
        project_id=project_id,
    )

    return {
        **paper_identity(paper),
        **result,
    }


async def extract_saved_paper_results(
    paper_title: str,
    project_id: int,
    focus: str | None = None,
) -> dict:
    paper, error = await resolve_saved_paper(
        paper_title=paper_title,
        project_id=project_id,
    )

    if error:
        return error

    result = extract_paper_results(
        paper_id=paper.id,
        project_id=project_id,
        focus=focus,
    )

    return {
        **paper_identity(paper),
        **result,
    }


async def compare_saved_papers(
    paper_titles: list[str],
    project_id: int,
) -> dict:
    papers, error = await resolve_saved_papers(
        paper_titles=paper_titles,
        project_id=project_id,
    )

    if error:
        return error

    result = compare_papers(
        paper_ids=[paper.id for paper in papers],
        project_id=project_id,
    )

    return {
        "papers": [paper_identity(paper) for paper in papers],
        **result,
    }


async def synthesize_saved_literature(
    paper_titles: list[str],
    topic: str,
    project_id: int,
) -> dict:
    papers, error = await resolve_saved_papers(
        paper_titles=paper_titles,
        project_id=project_id,
    )

    if error:
        return error

    result = synthesize_literature(
        paper_ids=[paper.id for paper in papers],
        project_id=project_id,
        topic=topic,
    )

    return {
        "papers": [paper_identity(paper) for paper in papers],
        **result,
    }




# --------------------------------------------------
# 5. Tool registry
# --------------------------------------------------

class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(self, name, function):
        self.tools[name] = function

    def get(self, name):
        return self.tools.get(name)

    def all_tools(self):
        return list(self.tools.values())


tool_registry = ToolRegistry()

PROJECT_SCOPED_TOOLS = {
    "search_and_save_papers",
    "answer_paper_question",
    "summarize_saved_paper",
    "extract_saved_paper_results",
    "compare_saved_papers",
    "synthesize_saved_literature",
}

tool_registry.register(
    "search_and_save_papers",
    search_and_save_papers,
)

tool_registry.register(
    "answer_paper_question",
    answer_paper_question,
)

tool_registry.register(
    "summarize_saved_paper",
    summarize_saved_paper,
)

tool_registry.register(
    "extract_saved_paper_results",
    extract_saved_paper_results,
)

tool_registry.register(
    "compare_saved_papers",
    compare_saved_papers,
)

tool_registry.register(
    "synthesize_saved_literature",
    synthesize_saved_literature,
)
# --------------------------------------------------
# 6. OpenAI tool definitions
# --------------------------------------------------

OPENAI_TOOLS = [
    {
        "type": "function",
        "name": "search_and_save_papers",
        "description": (
            "Search arXiv, save matching papers to the current project, "
            "and ingest their PDFs. Use this only when the user wants a "
            "paper discovery search or wants papers added as a list. For "
            "a paper needed by a summary, question, extraction, comparison, "
            "or synthesis, call that single content tool directly; it will "
            "find, save, ingest, and reload any missing paper before using "
            "the tool. For a named paper, use search_mode='title' and pass "
            "the title exactly as the user gave it. Do not append authors, "
            "dates, or related keywords. Use search_mode='topic' for a broad "
            "discovery query. Keep limit between 1 and 10. Each item reports "
            "whether its PDF is indexed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The exact paper title for a title search, or a concise "
                        "research topic for a topic search."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of papers to retrieve (1-10).",
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["title", "topic"],
                    "description": (
                        "Use 'title' for one named paper and 'topic' for discovery."
                    ),
                },
            },
            "required": ["query", "limit", "search_mode"],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "answer_paper_question",
        "description": (
            "Answer one specific question about one saved paper using "
            "retrieved PDF passages. Use this for a targeted question "
            "such as a method, architecture, equation, assumption, one "
            "reported number, or a limitation. Do not use this to "
            "summarize a whole paper, dump experimental tables, compare "
            "multiple papers, or perform a discovery search. If the named "
            "paper is missing, this tool finds it, saves it, ingests its PDF, "
            "and reloads it from the project before retrieval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question to answer from the paper.",
                },
                "paper_title": {
                    "type": "string",
                    "description": (
                        "The exact title from project metadata when it is "
                        "the requested work. If the work is not saved, use "
                        "its canonical title when a shorthand or description "
                        "identifies it unambiguously; never substitute a "
                        "similar saved paper. The tool will find and save a "
                        "missing paper. Never ask for a database ID."
                    )
                }
            },
            "required": ["question", "paper_title"],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "summarize_saved_paper",
        "description": (
            "Write a structured whole-paper summary of one paper. "
            "Use this when the user asks to summarize a paper, wants a "
            "TL;DR, or asks what a paper is about overall. Do not use "
            "this for a specific factual question, results extraction, "
            "multi-paper comparison, or literature synthesis. If the named "
            "paper is missing, find it, save it, ingest its PDF, and reload "
            "it from the project before summarizing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_title": {
                    "type": "string",
                    "description": (
                        "The canonical title of the paper selected from "
                        "the current project's saved paper metadata."
                    ),
                },
            },
            "required": ["paper_title"],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "extract_saved_paper_results",
        "description": (
            "Extract datasets, metrics, baselines, and reported "
            "experimental numbers from one paper. Use this when "
            "the user wants the results section, tables, or a list of "
            "metrics. Do not use this for a whole-paper summary, a "
            "single targeted question, or a multi-paper comparison. "
            "If the named paper is missing, find it, save it, ingest its "
            "PDF, and reload it from the project before extraction. "
            "If no extra focus is needed, pass focus as an empty string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_title": {
                    "type": "string",
                    "description": (
                        "The canonical title of the paper selected from "
                        "the current project's saved paper metadata."
                    ),
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "Optional extra focus, such as a dataset, "
                        "metric, or experiment name. Use an empty "
                        "string when no extra focus is needed."
                    ),
                },
            },
            "required": ["paper_title", "focus"],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "compare_saved_papers",
        "description": (
            "Compare two or more papers side by side. Use this "
            "when the user names specific papers and asks how they "
            "differ, which method is stronger, or what each one does "
            "relative to the others. Do not use this for a single-paper "
            "question, a whole-paper summary, or a topic-level "
            "literature review. Pass the titles or identifiers, including "
            "titles not yet saved in the project. This tool finds, saves, "
            "ingests, and reloads each missing paper before comparison. "
            "Do not call the search tool first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Canonical saved titles of the papers to compare. "
                        "Include at least two titles."
                    ),
                },
            },
            "required": ["paper_titles"],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "synthesize_saved_literature",
        "description": (
            "Synthesize a topic across two or more papers, "
            "including agreement, disagreement, and open gaps. Use this "
            "for related-work or literature-review requests. Do not use "
            "this when the user wants a direct pairwise comparison of "
            "named papers; use compare_saved_papers instead. Do not use "
            "this for a single paper. Missing named papers are found, saved, "
            "ingested, and reloaded before the tool runs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Canonical saved titles of the papers to synthesize. "
                        "Include at least two titles."
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "The research topic or question to synthesize."
                    ),
                },
            },
            "required": ["paper_titles", "topic"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# --------------------------------------------------
# 7. OpenAI agent call
# --------------------------------------------------
SYSTEM_PROMPT = """
You are the ResearchOS research assistant.

Help the user research academic papers in the current project. Use the
conversation history to resolve follow-ups, but decide the action from the
latest user message. Never invent paper findings, citations, or metadata.
Never ask for a database ID. Paper-specific claims must come from a
content tool result, not general knowledge or an abstract.

Choose exactly one action and call at most one tool for a user turn.
Never call multiple tools together, never call a search tool before a
content tool, and never switch to a second tool after the first tool
returns. The content tools own their complete preparation path: they
resolve each requested paper in this project; if a paper is missing, they
search arXiv for that specific title or ID, save the matching paper,
ingest its PDF, reload the saved database record, and only then retrieve
passages and run the tool. If identification or ingestion fails, report
that result; do not run a second search or make unsupported claims.

Route the request to one content tool:
- summarize_saved_paper for a whole-paper overview, summary, or TL;DR.
- extract_saved_paper_results for reported tables, metrics, datasets,
  baselines, or experimental results from one paper.
- answer_paper_question for one focused question about one paper.
- compare_saved_papers for a direct comparison of two or more named papers.
- synthesize_saved_literature for a topic-level literature review across
  two or more papers.

These actions are mutually exclusive for one turn. Do not summarize when
the user asks a focused question; do not extract results for a general
summary; do not synthesize when the user requests a direct comparison;
and do not compare when the user requests a topic-level literature review.
If the user only wants to discover or add a list of papers, call
search_and_save_papers and do not invoke a content tool in the same turn.

For questions about the user's interests, expertise, or background, or
about the current project's description and saved-paper list, answer from
the supplied user and project context without calling any tool. Do not
search for papers just to list papers already saved in the project.

For a follow-up that refers to "the last paper", "that paper", "it", or
similar wording, resolve the paper from the recent session history and the
current project's saved-paper metadata. Use summarize_saved_paper with the
resolved title for a whole-paper explanation or summary. Use
answer_paper_question for a focused question, rewriting vague pronouns into
a specific question. Do not send a phrase like "the last paper" to arXiv.
If recent chat history does not identify one paper unambiguously, ask which
paper the user means instead of calling a paper tool or searching.

For a named-paper search, set search_mode to "title" and pass the user's
title exactly. Do not append an author, year, field, or related keyword.
For broad discovery, set search_mode to "topic" and use a concise topic.
Use canonical titles from project context when available. For papers not
yet saved, pass the exact title or arXiv ID supplied by the user. When the
user identifies a well-known paper by a standard abbreviation or a clear
description rather than its full title, expand it to its canonical title
only when the identity is unambiguous. Do not substitute a similar saved
paper or send a topic description to a title lookup when the work cannot be
identified confidently; ask the user to clarify instead.

If the requested paper is ambiguous, ask which paper the user means and
include the candidate titles from project context. If no tool is needed,
answer directly.
"""


def preferred_tool_for_question(
    question: str,
    project_context: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> str | None:
    """Return a single high-confidence route for explicit user intents."""
    text = (question or "").casefold()
    follow_up_title = resolve_recent_paper_reference(
        question,
        conversation_history or [],
        project_context or {},
    )

    # If the user refers to a paper from earlier in the chat, only route it
    # when the referent can be resolved from that chat. The caller will let
    # the model ask a clarification question otherwise; it must not search
    # arXiv for a phrase such as "the last paper".
    if is_paper_reference_follow_up(question, project_context) and not follow_up_title:
        return None

    if re.search(
        r"\b(literature review|literature survey|related work|"
        r"synthesi[sz]e?|synthesis|state of the art)\b",
        text,
    ):
        return "synthesize_saved_literature"

    if re.search(
        r"\b(compare|comparison|contrast|versus|vs\.?|differences?|"
        r"side[-\s]?by[-\s]?side)\b",
        text,
    ) or re.search(r"\bdifference between\b", text):
        return "compare_saved_papers"

    if re.search(
        r"\b(extract|results?|metrics|datasets|"
        r"benchmarks|baselines|ablation|tables?|performance|quantitative|scores?)\b",
        text,
    ):
        return "extract_saved_paper_results"

    if re.search(
        r"\b(summarize|summarise|summary|overview|tl\s*;?\s*dr)\b",
        text,
    ):
        return "summarize_saved_paper"

    if (
        follow_up_title
        and is_paper_reference_follow_up(question, project_context)
        and re.search(r"\b(?:explain|walk me through|describe|tell me about)\b", text)
        and not re.search(
            r"\b(?:method|architecture|equation|assumption|finding|result|"
            r"contribution|contributions|approach|idea|motivation|experiment|"
            r"limitation|why|how|what does|what is|which|when|where|compare)\b",
            text,
        )
    ):
        # A broad explanation is a whole-paper overview. Focused questions
        # continue to the evidence-backed question tool below.
        return "summarize_saved_paper"

    # Requests to explain or walk through a singular paper are content
    # requests even when they describe it with a topic phrase such as
    # "the GCN paper for semi-supervised learning". They must not fall
    # through to broad discovery and ingest a list of related papers.
    if re.search(
        r"\b(explain|walk me through|describe|what is|what does|how does|"
        r"why does|tell me about)\b.{0,100}\b(?:the\s+)?"
        r"(?:paper|article|study|work)\b",
        text,
    ):
        return "answer_paper_question"

    if follow_up_title and is_paper_reference_follow_up(question, project_context):
        return "answer_paper_question"

    if re.search(
        r"\b(find|search for|discover|recommend|add|save|look for)\b"
        r".{0,50}\b(papers?|articles?)\b",
        text,
    ) or re.search(r"\b(papers|articles)\b.{0,35}\b(on|about|for)\b", text):
        return "search_and_save_papers"

    normalized_question = normalize_title(question)
    for paper in (project_context or {}).get("papers", []):
        title = normalize_title(paper.get("title") or "")
        if title and title in normalized_question:
            return "answer_paper_question"

    return None


def is_paper_reference_follow_up(
    question: str,
    project_context: dict | None = None,
) -> bool:
    """Whether a research request uses a pronoun for an earlier paper."""
    text = (question or "").casefold()
    normalized_question = normalize_title(question)
    if any(
        normalize_title(paper.get("title") or "") in normalized_question
        for paper in (project_context or {}).get("papers", [])
        if normalize_title(paper.get("title") or "")
    ):
        # A pronoun-like token may simply be part of an explicitly named title.
        return False
    has_research_action = re.search(
        r"\b(?:summari[sz]e|summary|overview|explain|walk me through|"
        r"describe|tell me|what|why|how|compare|contrast|extract|"
        r"results?|metrics|findings?)\b",
        text,
    )
    has_paper_reference = re.search(
        r"\b(?:last|previous|earlier|recently discussed|that|this|same)\s+"
        r"(?:discussed\s+)?(?:paper|article|study|work)\b|"
        r"\b(?:it|its)\b|\b(?:one|paper) we (?:just )?(?:discussed|talked about)\b",
        text,
    )
    return bool(has_research_action and has_paper_reference)


def is_paper_reference_value(value: str) -> bool:
    """Whether a tool argument still contains a vague paper placeholder."""
    return bool(
        re.fullmatch(
            r"\s*(?:(?:the\s+)?(?:last|previous|earlier|recently discussed)\s+"
            r"(?:discussed\s+)?(?:paper|article|study|work)(?:\s+we\s+"
            r"(?:just\s+)?(?:discussed|talked about))?|"
            r"(?:that|this|the same)(?:\s+(?:paper|article|study|work))?|"
            r"it|its|(?:the\s+)?(?:one|paper) we (?:just )?"
            r"(?:discussed|talked about))\s*",
            str(value),
            re.IGNORECASE,
        )
    )


def resolve_recent_paper_reference(
    question: str,
    conversation_history: list[dict],
    project_context: dict,
) -> str | None:
    """Resolve a paper pronoun to the latest canonical title in chat history."""
    if not is_paper_reference_follow_up(question, project_context):
        return None

    papers = project_context.get("papers", []) or []
    for message in reversed(conversation_history):
        content = str(message.get("content") or "")
        normalized_content = normalize_title(content)
        matches = []
        for paper in papers:
            title = (paper.get("title") or "").strip()
            normalized_title = normalize_title(title)
            if not title or not normalized_title:
                continue
            position = content.casefold().rfind(title.casefold())
            if position < 0:
                position = normalized_content.rfind(normalized_title)
            if position >= 0:
                matches.append((position, title))
        if matches:
            return max(matches, key=lambda item: item[0])[1]

    return None


def rewrite_paper_reference(question: str, paper_title: str) -> str:
    """Replace vague paper references before using the question for retrieval."""
    rewritten = question or ""
    rewritten = re.sub(
        r"\b(?:the\s+)?(?:last|previous|earlier|recently discussed)\s+"
        r"(?:discussed\s+)?(?:paper|article|study|work)\b",
        paper_title,
        rewritten,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"\b(?:that|this|the same)\s+(?:paper|article|study|work)\b",
        paper_title,
        rewritten,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(r"\bits\b", f"{paper_title}'s", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bit\b", paper_title, rewritten, flags=re.IGNORECASE)
    return rewritten


def is_context_only_question(
    question: str,
    preferred_tool: str | None = None,
) -> bool:
    """Identify context questions that should bypass every research tool."""
    text = (question or "").casefold()

    profile_question = re.search(
        r"\b(?:what do you know about me|what do you think (?:of|about) me|"
        r"describe me|profile me|what are my (?:research )?interests?|"
        r"what am i interested in|(?:explain|describe|summari[sz]e|tell me about) "
        r"(?:what )?my (?:research )?(?:interests?|expertise|background|profile)|"
        r"my (?:research )?(?:interests?|expertise|background|profile))\b",
        text,
    )
    if profile_question:
        return True

    # Recaps of chat history are answered from the supplied conversation
    # context. They are distinct from a paper follow-up such as "summarize it".
    if re.search(
        r"\b(?:recap|recapitulate|summari[sz]e our (?:(?:recent|last|previous) )?(?:chat|conversation|discussion)|"
        r"summari[sz]e what we (?:talked|discussed)(?: about)?|"
        r"summari[sz]e (?:the )?(?:last|previous|recent) (?:chat|conversation|discussion)|"
        r"what (?:did|have) we (?:talked|discussed) about|"
        r"what was our (?:last|previous) discussion)\b",
        text,
    ):
        return True

    paper_list_question = re.search(
        r"\b(?:list|show)\b.{0,50}\b(?:papers?|articles?)\b|"
        r"\bhow many\b.{0,50}\b(?:papers?|articles?)\b.{0,40}\b"
        r"(?:saved|in (?:this|the|my|our) (?:project|workspace|library))\b|"
        r"\b(?:what|which)\s+(?:papers?|articles?)\b.{0,60}\b"
        r"(?:in|saved|available|belong|do we have|have we saved)\b|"
        r"\b(?:papers?|articles?)\b.{0,40}\b(?:saved|in (?:this|the|my|our) "
        r"(?:project|workspace|library))\b",
        text,
    )
    explicit_search = bool(
        re.search(r"\b(?:find|search for|discover|recommend|add|save|look for)\b", text)
    )
    if paper_list_question and not explicit_search:
        return True

    project_description_question = re.search(
        r"\b(?:what is|what's|describe|tell me about|explain|summari[sz]e|"
        r"give me (?:an )?overview of)\b.{0,45}\b(?:project|workspace)\b|"
        r"\b(?:project|workspace)\s+(?:description|overview|purpose|goal)\b",
        text,
    )
    if project_description_question and not re.search(
        r"\b(?:paper|article|study|compare|comparison|extract|results?|metrics)\b",
        text,
    ):
        return True

    # A clear research action takes precedence over mentioning the project.
    if preferred_tool:
        return False

    if (
        re.search(r"\b(?:project|workspace)\b", text)
        and re.search(r"\b(?:papers?|articles?)\b", text)
    ):
        return True

    if re.search(
        r"\b(?:what is|what's|describe|tell me about|what do you know about) "
        r"(?:this|the|my|our) (?:project|workspace)\b",
        text,
    ):
        return True

    return False


async def ask_openai(contents, tool_choice="auto"):
    response = await client.responses.create(
        model="gpt-5.6-luna",
        instructions=SYSTEM_PROMPT,
        input=contents,
        tools=OPENAI_TOOLS,
        tool_choice=tool_choice,
        parallel_tool_calls=False,
    )

    return response


# --------------------------------------------------
# 8. Structured final response
# --------------------------------------------------


async def generate_final_response(
    question,
    conversation_history,
    agent_answer,
    tool_results,
    project_context,
    user_context,
):
    prompt = (
        "You are the final response generator for ResearchOS.\n\n"
        "Answer the user's latest question using the information below.\n\n"
        "CONVERSATION HISTORY:\n"
        f"{json.dumps(conversation_history, default=str)}\n\n"
        "LATEST QUESTION:\n"
        f"{question}\n\n"
        "AGENT'S RESPONSE:\n"
        f"{agent_answer}\n\n"
        "PROJECT CONTEXT:\n"
        f"{json.dumps(project_context, default=str)}\n\n"
        f"User preferences and expertise:\n"
        f"{json.dumps(user_context or {}, ensure_ascii=False, default=str)}\n\n"
        "RESULTS FROM TOOLS:\n"
        f"{json.dumps(tool_results, default=str)}\n\n"
        "Instructions:\n"
        "- Answer the latest question directly.\n"
        "- For general questions about the user's interests, expertise, "
        "project description, saved-paper list, or recent conversation, "
        "use the supplied context directly and do not invent missing details.\n"
        "- If a paper follow-up cannot be tied to one title in recent chat "
        "history, ask the user which paper they mean; do not search using "
        "a pronoun or a phrase such as 'the last paper'.\n"
        "- Format mathematical expressions with TeX delimiters: use "
        "$...$ inline and $$...$$ on separate lines for display equations. "
        "Do not leave raw TeX commands un-delimited.\n"
        "- Use project context for its description, saved-paper listing, "
        "and paper identity. Do not treat abstracts as evidence about "
        "paper contents.\n"
        "- Do not invent paper metadata, findings, or citations.\n"
        "- Copy paper metadata only from search results or the "
        "project paper list. If a field is unknown, use null or [].\n"
        "- Put papers in the papers list only when they were returned "
        "by search or used by a grounded tool. Otherwise return [].\n"
        "- If a tool returned summary, results, comparison, synthesis, "
        "or answer with grounded=true, use that content as the main "
        "answer and keep its headings. Do not mix in extra claims.\n"
        "- If a tool has needs_clarification=true, ask the user which "
        "paper they mean and list candidate_titles. Do not guess.\n"
        "- If ingestion_status is no_pdf_url or failed, state that the "
        "paper could not be indexed and do not imply its contents were read.\n"
        "- If a content tool returns an error or grounded=false, relay the "
        "limitation clearly and do not claim the tool completed.\n"
        "- Treat a successful content tool result as already prepared: any "
        "new paper was saved and indexed before retrieval. Do not describe "
        "or invent a separate search result unless one is present.\n"
        "- If no papers were searched, do not say that no answer is "
        "possible for that reason alone.\n"
    )
    if has_blocking_tool_result(tool_results):
        prompt += (
            "\nIMPORTANT: No grounded paper evidence is available. "
            "Do not answer paper-specific questions from general "
            "knowledge. Explain the limitation or ask for "
            "clarification.\n"
        )
    response = await client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "research_response",
                "schema": ResearchResponse.model_json_schema(),
                "strict": True,
            }
        },
    )

    return ResearchResponse.model_validate_json(
        response.output_text
    )


# --------------------------------------------------
# 9. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Hello ResearchOS"
    }


def save_chat_message(session_id: int, role: str, content: str):
    db = SessionLocal()

    try:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
        )

        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()



def load_recent_messages(session_id: int, limit: int = 10):
    db = SessionLocal()

    try:
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(
                ChatMessage.created_at.desc(),
                ChatMessage.id.desc(),
            )
            .limit(limit)
            .all()
        )

        # The database query returns newest first.
        # Reverse the list to restore chronological order.
        messages.reverse()

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

    finally:
        db.close()


def load_project_context(project_id: int) -> dict:
    db = SessionLocal()

    try:
        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise HTTPException(
                status_code=404,
                detail="Project not found",
            )

        papers = (
            db.query(Paper)
            .join(
                ProjectPaper,
                ProjectPaper.paper_id == Paper.id,
            )
            .filter(ProjectPaper.project_id == project_id)
            .order_by(ProjectPaper.created_at.desc())
            .all()
        )

        return {
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
            },
            "papers": [
                {
                    "id": paper.id,
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "paper_url": paper.paper_url,
                }
                for paper in papers
            ],
        }

    finally:
        db.close()



# --------------------------------------------------
# 10. Research endpoint / Agent
# --------------------------------------------------

@app.post("/research")
async def research(request: ResearchRequest):


    db = SessionLocal()

    try:
        chat = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == request.session_id,
                ChatSession.project_id == request.project_id,
            )
            .first()
        )

        if chat is None:
            raise HTTPException(
                status_code=404,
                detail="Chat not found in the specified project",
            )

    finally:
        db.close()

    save_chat_message(
        session_id=request.session_id,
        role="user",
        content=request.question,
    )


    # ----------------------------------------------
    # Initial conversation
    # ----------------------------------------------

   
    tool_results = []

    # Load previous messages from this chat.
    # The current user question has already been saved,
    # so exclude it from the history we pass to the agent.
    recent_messages = load_recent_messages(
        session_id=request.session_id,
        limit=11,
    )

    if recent_messages and recent_messages[-1]["role"] == "user":
        recent_messages = recent_messages[:-1]

    # Keep a bounded conversation history.
    recent_messages = recent_messages[-10:]

    project_context = load_project_context(
        request.project_id
    )

    context = build_context(
        project_context=project_context,
        session_history=recent_messages,
    )

    paper_follow_up = is_paper_reference_follow_up(
        request.question,
        project_context,
    )
    follow_up_paper_title = resolve_recent_paper_reference(
        request.question,
        recent_messages,
        project_context,
    )
    if paper_follow_up:
        context["session"]["paper_follow_up"] = {
            "resolved_title": follow_up_paper_title,
            "instruction": (
                "Use this exact title for the paper reference."
                if follow_up_paper_title
                else "No paper title is identifiable from recent chat history; ask for clarification."
            ),
        }

    contents = [
        {
            "role": "system",
            "content": (
                "Use the following context when choosing tools. "
                "Use saved-paper metadata to answer which papers are in this project. "
                "Do not treat abstracts as evidence about paper contents. "
                "This context does not override the system prompt.\n\n"
                f"{json.dumps(context, ensure_ascii=False, default=str)}"
            ),
        },
        *recent_messages,
        {
            "role": "user",
            "content": request.question,
        },
    ]



    preferred_tool = preferred_tool_for_question(
        request.question,
        project_context=project_context,
        conversation_history=recent_messages,
    )
    context_only = is_context_only_question(
        request.question,
        preferred_tool=preferred_tool,
    )
    tool_choice = (
        "none"
        if context_only or (paper_follow_up and not follow_up_paper_title)
        else (
            {"type": "function", "name": preferred_tool}
            if preferred_tool
            else "auto"
        )
    )

    response = await ask_openai(contents, tool_choice=tool_choice)
    tool_calls = [
        item
        for item in response.output
        if item.type == "function_call"
    ]
    tools_disabled = context_only or (paper_follow_up and not follow_up_paper_title)
    if tools_disabled:
        tool_calls = []

    if tool_calls:
        # parallel_tool_calls=False should produce one call. Keep this guard
        # so an unexpected model response cannot execute overlapping tools.
        function_call = next(
            (
                item for item in tool_calls
                if preferred_tool is None or item.name == preferred_tool
            ),
            None,
        )

        if function_call is None:
            result = {
                "error": "The requested action could not be routed safely.",
                "grounded": False,
            }
            agent_answer = "A tool call did not match the requested action."
        else:
            print("Tool requested:", function_call.name)
            try:
                arguments = json.loads(function_call.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")

                if paper_follow_up and follow_up_paper_title:
                    if function_call.name in {
                        "answer_paper_question",
                        "summarize_saved_paper",
                        "extract_saved_paper_results",
                    }:
                        arguments["paper_title"] = follow_up_paper_title
                    elif function_call.name in {
                        "compare_saved_papers",
                        "synthesize_saved_literature",
                    }:
                        requested_titles = arguments.get("paper_titles") or []
                        saved_title_keys = {
                            normalize_title(paper.get("title") or "")
                            for paper in project_context.get("papers", [])
                        }
                        reference_values = [
                            title for title in requested_titles
                            if (
                                normalize_title(title) not in saved_title_keys
                                and is_paper_reference_value(title)
                            )
                        ]
                        for reference_value in reference_values:
                            requested_titles[requested_titles.index(reference_value)] = (
                                follow_up_paper_title
                            )
                        if (
                            not reference_values
                            and len(requested_titles) == 1
                            and normalize_title(requested_titles[0])
                            != normalize_title(follow_up_paper_title)
                        ):
                            requested_titles.insert(0, follow_up_paper_title)
                        arguments["paper_titles"] = list(
                            dict.fromkeys(requested_titles)
                        )

                    if function_call.name == "answer_paper_question":
                        arguments["question"] = rewrite_paper_reference(
                            request.question,
                            follow_up_paper_title,
                        )

                tool = tool_registry.get(function_call.name)
                if tool is None:
                    raise ValueError(f"Unknown tool: {function_call.name}")

                if function_call.name in PROJECT_SCOPED_TOOLS:
                    result = tool(
                        **arguments,
                        project_id=request.project_id,
                    )
                else:
                    result = tool(**arguments)

                if inspect.isawaitable(result):
                    result = await result

            except Exception as exc:
                result = {
                    "error": str(exc),
                    "grounded": False,
                }

            agent_answer = response.output_text or "The requested research action completed."

        tool_results.append(result)
        print("Tool result:", result)
    else:
        agent_answer = response.output_text

    final_response = await generate_final_response(
        question=request.question,
        conversation_history=recent_messages,
        agent_answer=agent_answer,
        tool_results=tool_results,
        project_context=project_context,
        user_context=context["user"],
    )

    save_chat_message(
        session_id=request.session_id,
        role="assistant",
        content=final_response.answer,
    )

    return {
        "answer": final_response.answer,
        "papers": [paper.model_dump() for paper in final_response.papers],
        "iterations": 1,
    }


# --------------------------------------------------
# 11. Projects
# --------------------------------------------------

@app.get("/projects")
def get_projects():

    db = SessionLocal()

    projects = db.query(Project).all()

    db.close()

    return {
        "projects": [
            {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "created_at": project.created_at,
            }
            for project in projects
        ]
    }

@app.get("/projects/{project_id}/papers")
def get_project_papers(project_id: int):
    db = SessionLocal()

    try:
        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise HTTPException(
                status_code=404,
                detail="Project not found",
            )

        papers = (
            db.query(Paper)
            .join(
                ProjectPaper,
                ProjectPaper.paper_id == Paper.id,
            )
            .filter(ProjectPaper.project_id == project_id)
            .order_by(Paper.created_at.desc())
            .all()
        )

        return {
            "project_id": project_id,
            "papers": [
                {
                    "id": paper.id,
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "published": (
                        paper.published.isoformat()
                        if paper.published
                        else None
                    ),
                    "paper_url": paper.paper_url,
                    "pdf_url": paper.pdf_url,
                }
                for paper in papers
            ],
        }

    finally:
        db.close()

@app.post("/projects")
def create_project(
    project_data: ProjectCreate
):

    db = SessionLocal()

    project = Project(
        name=project_data.name,
        description=project_data.description,
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    db.close()

    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "created_at": project.created_at,
    }

@app.post(
    "/projects/{project_id}/chats",
    response_model=ChatSessionResponse,
    status_code=201,
)
def create_chat(
    project_id: int,
    chat_data: ChatSessionCreate,
):
    db = SessionLocal()

    try:
        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise HTTPException(
                status_code=404,
                detail="Project not found",
            )

        chat = ChatSession(
            project_id=project_id,
            title=chat_data.title,
        )

        db.add(chat)
        db.commit()
        db.refresh(chat)

        return chat

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

@app.get(
    "/projects/{project_id}/chats",
    response_model=list[ChatSessionResponse],
)
def get_project_chats(project_id: int):
    db = SessionLocal()

    try:
        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise HTTPException(
                status_code=404,
                detail="Project not found",
            )

        chats = (
            db.query(ChatSession)
            .filter(ChatSession.project_id == project_id)
            .order_by(ChatSession.created_at.desc())
            .all()
        )

        return chats

    finally:
        db.close()

@app.get(
    "/chats/{chat_id}/messages",
    response_model=list[ChatMessageResponse],
)
def get_chat_messages(chat_id: int):
    db = SessionLocal()

    try:
        chat = (
            db.query(ChatSession)
            .filter(ChatSession.id == chat_id)
            .first()
        )

        if chat is None:
            raise HTTPException(
                status_code=404,
                detail="Chat not found",
            )

        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == chat_id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
            .all()
        )

        return messages

    finally:
        db.close()


@app.post(
    "/chats/{chat_id}/messages",
    response_model=ChatMessageResponse,
    status_code=201,
)
def create_chat_message(
    chat_id: int,
    message_data: ChatMessageCreate,
):
    content = message_data.content.strip()

    if not content:
        raise HTTPException(
            status_code=422,
            detail="Message content cannot be empty",
        )

    db = SessionLocal()

    try:
        chat = (
            db.query(ChatSession)
            .filter(ChatSession.id == chat_id)
            .first()
        )

        if chat is None:
            raise HTTPException(
                status_code=404,
                detail="Chat not found",
            )

        message = ChatMessage(
            session_id=chat_id,
            role="user",
            content=content,
        )

        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
