from fastapi import FastAPI,HTTPException
from httpcore2 import request
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
import json
import re

from dotenv import load_dotenv

from app.database import SessionLocal
from app.models import Project,Paper,ProjectPaper,ChatSession,ChatMessage
from app.services.arxiv import search_papers
from app.services.papers import save_papers
from app.schemas import ProjectCreate, ResearchResponse,ResearchRequest,ChatSessionCreate,ChatSessionResponse,ChatMessageCreate,ChatMessageResponse


load_dotenv()


# --------------------------------------------------
# 1. FastAPI application
# --------------------------------------------------

app = FastAPI()


# --------------------------------------------------
# 2. OpenAI client
# --------------------------------------------------

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# 4. Tools
# --------------------------------------------------

def get_weather(city: str):
    """
    Return the weather for a city.
    """
    return f"The weather in {city} is sunny."


def get_capital(country: str):
    """
    Return the capital of a country.
    """
    capitals = {
        "India": "New Delhi",
        "France": "Paris",
        "Japan": "Tokyo",
        "Germany": "Berlin",
        "United Kingdom": "London",
    }

    return capitals.get(
        country,
        f"I don't know the capital of {country}."
    )



def search_and_save_papers(
    project_id: int,
    query: str,
    limit: int = 5,
):
    # If the query contains an arXiv ID, prioritize the ID.
    match = re.search(
        r"(?i)(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)",
        query,
    )

    if match:
        query = match.group(1)

    print("Final arXiv search query:", query)

    papers = search_papers(
        query=query,
        limit=limit,
    )

    saved_papers = save_papers(papers, project_id)

    return [
        {
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
        for paper in saved_papers
    ]



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

tool_registry.register(
    "get_weather",
    get_weather,
)

tool_registry.register(
    "get_capital",
    get_capital,
)

tool_registry.register(
    "search_and_save_papers",
    search_and_save_papers,
)


# --------------------------------------------------
# 6. OpenAI tool definitions
# --------------------------------------------------

OPENAI_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get weather for.",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "get_capital",
        "description": "Get the capital of a country.",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "The country.",
                }
            },
            "required": ["country"],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "search_and_save_papers",
        "description": (
            "Search arXiv for academic papers and save the discovered "
            "papers to the ResearchOS database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The academic search query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of papers to retrieve.",
                },
            },
            "required": ["query", "limit"],
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

Your job is to help the user research academic topics.

You have access to research tools that retrieve information.

Use search_and_save_papers when you need to discover papers.

Do not repeatedly search for slightly different versions of the same query.

You are allowed only one search per iteration.

Do not search for the same query multiple times.

Once you have enough relevant information, stop using tools.

When presenting discovered papers, include their available metadata and links.
"""


async def ask_openai(contents):
    response = await client.responses.create(
        model="gpt-5.6-luna",
        instructions=SYSTEM_PROMPT,
        input=contents,
        tools=OPENAI_TOOLS,
    )

    return response


# --------------------------------------------------
# 8. Structured final response
# --------------------------------------------------

async def generate_final_response(papers):

    prompt = (
        "You are the final response generator for ResearchOS.\n\n"
        "Using the papers below, produce a structured response.\n"
        "Do not invent paper metadata.\n"
        "Use only the information provided.\n\n"
        f"Papers:\n{json.dumps(papers, default=str)}"
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


# --------------------------------------------------
# 10. Research endpoint / Agent
# --------------------------------------------------

@app.post("/research")
async def research(request: ResearchRequest):

    # ----------------------------------------------
    # Initial conversation
    # ----------------------------------------------

    tool_results = []

    contents = [
        {
            "role": "user",
            "content": request.question,
        }
    ]


    # ----------------------------------------------
    # Maximum number of agent iterations
    # ----------------------------------------------

    MAX_ITERATIONS = 2


    # ----------------------------------------------
    # Agent loop
    # ----------------------------------------------

    for iteration in range(MAX_ITERATIONS):

        print(
            f"\n--- Agent iteration "
            f"{iteration + 1} ---"
        )


        # ------------------------------------------
        # Ask OpenAI
        # ------------------------------------------

        response = await ask_openai(
            contents
        )


        # ------------------------------------------
        # Add model output to conversation
        # ------------------------------------------

        contents.extend(
            response.output
        )


        # ------------------------------------------
        # Find tool calls
        # ------------------------------------------

        tool_calls = []

        for item in response.output:

            if item.type == "function_call":

                tool_calls.append(item)


        # ------------------------------------------
        # No tool call
        # → Agent has finished
        # ------------------------------------------

        if not tool_calls:

            print("Agent finished.")

            final_response = await generate_final_response(
                tool_results
            )

            return {
                "answer": final_response.answer,
                "papers": [
                    paper.model_dump()
                    for paper in final_response.papers
                ],
                "iterations": iteration + 1,
            }


        # ------------------------------------------
        # Execute tools
        # ------------------------------------------

        for function_call in tool_calls:

            print(
                "Tool requested:",
                function_call.name
            )
            arguments = json.loads(function_call.arguments)
            print("Tool arguments:", arguments)

            tool = tool_registry.get(function_call.name)

            if tool is None:
                result = {"error": f"Unknown tool: {function_call.name}"}
            else:
                try:
                    if function_call.name == "search_and_save_papers":
                        result = tool(
                            **arguments,
                            project_id=request.project_id,
                        )
                    else:
                        result = tool(**arguments)

                except Exception as e:
                    result = {"error": str(e)}

            tool_results.append(result)


            print("Tool result:", result)

            if isinstance(result, dict) and "error" in result:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "tool": function_call.name,
                        "message": str(result["error"]),
                    },
                )


            # --------------------------------------
            # Send result back to OpenAI
            # --------------------------------------

            contents.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": json.dumps(
                        result,
                        default=str,
                    ),
                }
            )


    # ----------------------------------------------
    # Maximum iterations reached
    # ----------------------------------------------

    return {
        "error": (
            "Agent reached maximum iterations."
        ),
        "iterations": MAX_ITERATIONS,
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
