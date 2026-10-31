# ResearchOS - Research Orchestration System

ResearchOS is a small research assistant for collecting papers and asking questions about them. It searches arXiv, saves papers to a project, reads their PDFs, and uses passages from those PDFs to answer paper-related requests.

This project was built to practice LLM tool calling, focused research actions, and a basic retrieval-augmented generation (RAG) pipeline.

## What the app does

- Creates research projects, each with its own paper list and saved chats.
- Finds papers using an arXiv ID, an exact title, or a broad research topic.
- Saves paper metadata and links a paper to the selected project.
- Ingests new PDFs when a content action needs them.
- Answers a focused question, summarizes one paper, extracts its reported results, compares papers, or synthesizes literature across papers.
- Uses recent chat and project information for context questions that do not need paper passages.
- Shows source papers and page information with grounded research answers.

## Architecture overview

```mermaid
flowchart LR
    User --> UI[Browser UI and action selector]
    UI --> API[FastAPI request endpoint]
    API --> Context[Last 10 chat messages and project metadata]
    API --> Router[Choose one callable tool]
    Router --> Search[arXiv search and paper saving]
    Search --> SQL[PostgreSQL metadata tables]
    SQL --> Download[Download and read a PDF]
    Download --> Chunks[Split text into chunks]
    Chunks --> Embeddings[OpenAI embeddings]
    Embeddings --> VectorDB[PostgreSQL with pgvector]
    Router --> Retrieval[Retrieve evidence for a paper and project]
    VectorDB --> Retrieval
    Retrieval --> Model[Generate an answer from excerpts]
    Model --> Chats[Save assistant response]
    Chats --> UI
```

### The request path

1. The user selects an action in the frontend and enters a request.
2. The frontend adds a short action prefix, such as `Summarize this paper:`, then sends the request, project ID, and chat ID to `POST /research`.
3. FastAPI loads project paper metadata and up to 10 earlier messages from the current chat. It resolves clear references such as “summarize it” when the recent chat identifies a paper.
4. The backend chooses one callable tool. For content requests about a paper that is not saved yet, the tool finds the paper, saves its metadata, and attempts PDF ingestion before retrieval.
5. The ingestion pipeline extracts page text, splits it into overlapping chunks, generates embeddings, and writes the chunks and metadata to pgvector.
6. The retriever searches only chunks matching both the selected paper ID and project ID. The chosen tool handler passes those excerpts to the model for the requested task.
7. The response and paper references are saved to the chat and returned to the browser.

General questions about the current project, saved papers, recent chat, or user context can be answered from supplied context without running paper RAG.

## Callable tools

The model is given six callable functions. Each one has a distinct purpose so the app can keep search separate from paper analysis.

| Tool | Purpose |
| --- | --- |
| `search_and_save_papers` | Search arXiv for a title or topic, save results to the current project, and try to index the PDFs. |
| `answer_paper_question` | Answer one focused question using retrieved passages from one paper. |
| `summarize_saved_paper` | Produce a structured, evidence-based summary of one paper. |
| `extract_saved_paper_results` | Extract datasets, metrics, baselines, and reported results. |
| `compare_saved_papers` | Compare evidence retrieved separately from two or more papers. |
| `synthesize_saved_literature` | Synthesize evidence from multiple papers around a topic. |

The frontend action selector lets the user choose the intended action before sending. The selector does not send a raw function ID; it makes the intent explicit in the request text. The backend then routes the request and constrains the model to the matching tool when possible. Only one tool call is allowed per turn. This simple selector was chosen after open-ended requests sometimes overlapped, such as summarization and results extraction.

The Python package `backend/app/tools/` contains the search/save and focused Q&A handlers, the summary/results/comparison/synthesis workflows, and shared paper-resolution, evidence, and generation helpers. `main.py` owns the API, model-facing schemas, routing, dispatch, and the thin wrappers for summary, results, comparison, and synthesis. A handler may perform several internal steps, such as retrieval followed by generation, while still being invoked through one tool call.

## Storage model

PostgreSQL stores the records needed by the app:

- `projects` stores project names and descriptions.
- `papers` stores arXiv IDs, titles, abstracts, authors, and links.
- `project_papers` connects saved papers to projects. A paper can be reused across projects.
- `chat_sessions` and `chat_messages` store conversations for each project.
- `user_profiles` stores user preferences, interests, and expertise context.

The pgvector table stores embedded PDF chunks. Each chunk carries its paper ID, project ID, arXiv ID, and page number. These fields let retrieval scope results and display sources.

## Tech stack

- **Backend:** Python, FastAPI, SQLAlchemy, Pydantic
- **LLM and embeddings:** OpenAI API
- **Paper discovery:** arXiv Atom API
- **PDF and RAG:** LlamaIndex, PDFReader, sentence splitting, vector retrieval
- **Database:** PostgreSQL and pgvector
- **Frontend:** HTML, CSS, and JavaScript
- **Local database setup:** Docker Compose

## Run it locally

You need Python 3.10 or later, Docker with Docker Compose, and an OpenAI API key.

1. From the project root, create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the backend dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```

3. Start PostgreSQL with pgvector:

   ```bash
   docker compose up -d postgres
   ```

4. Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   DATABASE_URL=postgresql://researchos:researchos@localhost:5432/researchos
   ```

   Keep this file local. `.gitignore` excludes it so API keys are not committed.

5. Create the relational tables and start the API:

   ```bash
   cd backend
   python create_tables.py
   uvicorn app.main:app --reload
   ```

6. Open the app at [http://localhost:8000/ui/](http://localhost:8000/ui/). The interactive API page is at [http://localhost:8000/docs](http://localhost:8000/docs).

The Docker volume keeps database data when the container stops. The vector store is initialized when it is first used.

## API endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /projects` | List projects. |
| `POST /projects` | Create a project. |
| `GET /projects/{project_id}/papers` | List papers saved in a project. |
| `POST /projects/{project_id}/chats` | Create a chat for a project. |
| `GET /projects/{project_id}/chats` | List chats in a project. |
| `GET /chats/{chat_id}/messages` | Load a chat history. |
| `POST /chats/{chat_id}/messages` | Save a user message. |
| `POST /research` | Route a research request, call a tool if needed, and save the answer. |

`POST /research` accepts a `question`, `project_id`, and `session_id`.

## Codebase map

- `backend/app/main.py` - FastAPI routes, OpenAI tool definitions, request routing, context loading, and tool dispatch.
- `backend/app/tools/search_and_save.py` - Searches arXiv, saves paper records to a project, and indexes available PDFs.
- `backend/app/tools/answer_paper_question.py` - Resolves a paper, ensures it is indexed, retrieves relevant excerpts, and answers one focused question.
- `backend/app/tools/paper_resolution.py` - Shared title and arXiv ID matching, saved-paper lookup, PDF readiness checks, and paper metadata formatting.
- `backend/app/tools/summarisation.py`, `results_extraction.py`, `paper_comparison.py`, and `literature_synthesis.py` - Retrieve evidence and generate summaries, results, comparisons, and literature syntheses.
- `backend/app/tools/evidence.py` and `generate.py` - Shared retrieval/source formatting and grounded answer-generation helpers.
- `backend/app/rag/` - PDF download/ingestion, vector-store setup, and filtered retrieval.
- `backend/app/services/` - arXiv search, paper persistence, and context assembly.
- `backend/app/models.py` and `backend/app/schemas.py` - Database tables and API data shapes.
- `frontend/` - Page structure, browser behavior, action selection, and responsive styling.
- `backend/create_tables.py` - Creates the relational database tables.
- `docker-compose.yml` - Runs the local PostgreSQL/pgvector container.

The interview guide in `output/pdf/` gives a short purpose and role for every code, configuration, and test file.

## Design choices

- **One tool per request:** The action selector clarifies intent, and `parallel_tool_calls=False` plus backend routing prevents a request from invoking several overlapping actions.
- **Separate title search from topic discovery:** A title request should find the named work. A broad topic request can return related work. Keeping these search modes separate prevents a paper that only cites a named paper from being mistaken for it.
- **Save metadata in SQL and chunks in pgvector:** Project and chat lists need exact database queries; paper passages need semantic similarity search.
- **Filter retrieval by paper and project:** This reduces the chance that evidence from another paper or workspace is used in an answer.
- **Ingest before content retrieval:** New papers are saved and indexed before a content tool asks the vector store for passages. A successful search result alone is not treated as evidence from the PDF.
- **Use context for context questions:** Recent conversation and project metadata are supplied directly when paper retrieval is unnecessary.
- **Keep one small service:** FastAPI serves the API and static frontend, with PostgreSQL as the local persistence layer. This kept the first version easier to run and understand.

## Challenges and current limits

- **Paper search accuracy:** A broad search once returned a paper that cited the requested work. The search path was tightened so title searches stay title-focused while topic discovery can search more broadly.
- **Overlapping actions:** Prompt-only selection sometimes confused similar tasks. We returned to a simple frontend action selector. A future version could improve automatic selection and evaluate it against sample requests.
- **Ingestion time:** PDF text must be extracted, chunked, embedded, and stored before it can be searched. This can make the first content request slower than later requests.
- **Unusual PDF text:** One PDF produced an ingestion error mentioning NUL (`0x00`) characters. The current pipeline does not sanitize extracted text or switch parsers; that remains a possible improvement.
- **Frontend edge cases:** Long chat scrolling and math rendering needed presentation fixes. These changes were kept in the frontend and did not change the research workflows.

This is a learning project, not a production-hardened service. There is no systematic tool-routing or retrieval benchmark, background ingestion worker, authentication layer, or deployment setup yet.
