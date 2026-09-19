from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import os

from dotenv import load_dotenv
import psycopg

from app.database import SessionLocal
from app.models import Project
from app.schemas import ProjectCreate

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():

    return psycopg.connect(
        DATABASE_URL
    )


# --------------------------------------------------
# 1. FastAPI application
# --------------------------------------------------

app = FastAPI()


# --------------------------------------------------
# 2. Gemini client
# --------------------------------------------------

client = genai.Client()


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
    get_weather
)

tool_registry.register(
    "get_capital",
    get_capital
)


# --------------------------------------------------
# 6. Gemini call
# --------------------------------------------------

async def ask_gemini(contents):

    response = await client.aio.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config={
            "tools": tool_registry.all_tools(),

            # We manually execute tools.
            "automatic_function_calling": {
                "disable": True
            },
        },
    )

    return response


# --------------------------------------------------
# 7. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Hello ResearchOS"
    }


# --------------------------------------------------
# 8. Research endpoint / Agent
# --------------------------------------------------

@app.post("/research")
async def research(request: ResearchRequest):

    # ----------------------------------------------
    # Initial conversation
    # ----------------------------------------------

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=request.question
                )
            ],
        )
    ]

    # ----------------------------------------------
    # Maximum number of agent iterations
    # ----------------------------------------------

    MAX_ITERATIONS = 10

    # ----------------------------------------------
    # Agent loop
    # ----------------------------------------------

    for iteration in range(MAX_ITERATIONS):

        print(
            f"\n--- Agent iteration "
            f"{iteration + 1} ---"
        )

        # ------------------------------------------
        # Ask Gemini
        # ------------------------------------------

        response = await ask_gemini(contents)

        # ------------------------------------------
        # Get model's response
        # ------------------------------------------

        model_content = response.candidates[0].content

        # Keep model response in conversation
        contents.append(model_content)

        # ------------------------------------------
        # Find tool calls
        # ------------------------------------------

        tool_calls = []

        for part in model_content.parts:

            if part.function_call:

                tool_calls.append(
                    part.function_call
                )

        # ------------------------------------------
        # No tool call
        # → Agent has finished
        # ------------------------------------------

        if not tool_calls:

            print("Agent finished.")

            return {
                "answer": response.text,
                "iterations": iteration + 1
            }

        # ------------------------------------------
        # Execute tools
        # ------------------------------------------

        for function_call in tool_calls:

            print(
                "Tool requested:",
                function_call.name
            )

            print(
                "Arguments:",
                function_call.args
            )

            # --------------------------------------
            # Check tool exists
            # --------------------------------------

            if function_call.name not in tool_registry.all_tools():

                result = {
                    "error": (
                        f"Unknown tool: "
                        f"{function_call.name}"
                    )
                }

            else:

                tool = tool_registry.get(function_call.name)

                # ----------------------------------
                # Execute tool
                # ----------------------------------

                try:

                    result = tool(
                        **function_call.args
                    )

                except Exception as e:

                    result = {
                        "error": str(e)
                    }

            print(
                "Tool result:",
                result
            )

            # --------------------------------------
            # Send result back to Gemini
            # --------------------------------------

            function_response_part = (
                types.Part.from_function_response(
                    name=function_call.name,
                    response={
                        "result": result
                    },
                )
            )

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        function_response_part
                    ],
                )
            )

    # ----------------------------------------------
    # Maximum iterations reached
    # ----------------------------------------------

    return {
        "error": (
            "Agent reached maximum iterations."
        ),
        "iterations": MAX_ITERATIONS
    }

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
def create_project(project_data: ProjectCreate):

    db = SessionLocal()

    project = Project(
        name=project_data.name,
        description=project_data.description
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