from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
import json

from dotenv import load_dotenv

from app.database import SessionLocal
from app.models import Project
from app.services.arxiv import search_papers
from app.services.papers import save_papers
from app.schemas import ProjectCreate, ResearchResponse


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
# 3. Request model
# --------------------------------------------------

class ResearchRequest(BaseModel):
    question: str


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


def search_and_save_papers(query: str, limit: int = 5):
    papers = search_papers(
        query=query,
        limit=limit,
    )

    saved_papers = save_papers(papers)

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

            arguments = json.loads(
                function_call.arguments
            )

            print(
                "Arguments:",
                arguments
            )


            # --------------------------------------
            # Find tool
            # --------------------------------------

            tool = tool_registry.get(
                function_call.name
            )


            if tool is None:

                result = {
                    "error": (
                        f"Unknown tool: "
                        f"{function_call.name}"
                    )
                }

            else:

                try:

                    result = tool(
                        **arguments
                    )

                except Exception as e:

                    result = {
                        "error": str(e)
                    }


            tool_results.append(result)


            print(
                "Tool result:",
                result
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