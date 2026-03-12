# RAG API (FastAPI transfer layer)

HTTP API used as a transfer layer between the RAG service and a .NET REST API.

## Prerequisites

**Run the Qdrant server first.** The RAG pipeline needs Qdrant for vector search (e.g. at `localhost:6333`). Start it before starting the API or testing through Swagger. For example:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or use your project’s DB bootstrap (e.g. `dev_tools/start_app_db.bat`). Without Qdrant, `/ask` will return an “incomplete” message when no chunks are found.

## Testing the API through Swagger

FastAPI serves **Swagger UI** so you can run the app and send requests from the browser.

1. **Start the API** (from repo root):

   ```bash
   uvicorn AI_api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Open Swagger UI** in your browser:

   ```
   http://localhost:8000/docs
   ```

3. **Try the endpoints:**
   - **GET /health** — Click “Try it out” → “Execute”. Response: `{"status": "ok"}`.
   - **POST /ask** — Click “Try it out”, set the request body to e.g. `{"question": "What is the main topic?"}` (optionally add `"history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`), then “Execute”. Response: `{"answer": "..."}`.

Alternative docs: **ReDoc** at `http://localhost:8000/redoc`. OpenAPI schema: `http://localhost:8000/openapi.json`.
