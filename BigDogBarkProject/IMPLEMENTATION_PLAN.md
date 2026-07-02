# BigDogBarkProject Implementation Plan

## 0. Goal

Build a new lightweight project named `BigDogBarkProject`.

The project must reuse the useful chat frontend from:

`E:\AI_agent_Learning\Stage2_学习工具调用、RAG 与记忆\SuperMew\frontend`

It must connect that frontend to the existing LangGraph search agent logic from:

`E:\AI_agent_Learning\Stage2_学习工具调用、RAG 与记忆\langgraph_search_agent.py`

Use Plan A:

- Keep a Vue/Vite frontend copied from SuperMew.
- Create a lightweight FastAPI backend.
- Implement only the interfaces the frontend needs.
- Leave RAG and document-management interfaces empty or mocked.
- Do not bring in PostgreSQL, Redis, Milvus, JWT complexity, or the full SuperMew backend.

## 1. Non-Negotiable Rules

1. Do not edit files outside `E:\AI_agent_Learning\BigDogBarkProject`.
2. Do not modify the original `SuperMew` project.
3. Do not modify the original `langgraph_search_agent.py`.
4. Do not implement real RAG.
5. Do not add PostgreSQL, Redis, Milvus, Docker, or database migrations.
6. Keep all session/auth data in memory for the first version.
7. The first successful milestone is: user opens the frontend, sends a message, receives streamed text from the LangGraph agent, and can stop generation.
8. If a step fails, stop and record the exact error before continuing.

## 2. Final Target Structure

Create this structure:

```text
BigDogBarkProject/
  IMPLEMENTATION_PLAN.md
  README.md
  .env.example
  backend/
    __init__.py
    app.py
    agent_adapter.py
    schemas.py
    memory_store.py
    rag_stub.py
  frontend/
    package.json
    package-lock.json
    index.html
    vite.config.ts
    tsconfig.json
    tsconfig.node.json
    src/
      ...
```

The `frontend/` directory should initially be copied from SuperMew frontend, then simplified.

## 3. Backend Scope

Create a small FastAPI backend. It must expose these routes:

### 3.1 Auth Routes

These are mock routes only.

`POST /auth/login`

Request body:

```json
{
  "username": "anything",
  "password": "anything"
}
```

Response:

```json
{
  "access_token": "dev-token",
  "username": "anything",
  "role": "user"
}
```

`POST /auth/register`

Use the same behavior as login. If `role` is provided, return it. If not, return `user`.

`GET /auth/me`

Response:

```json
{
  "username": "dev-user",
  "role": "user"
}
```

Do not validate passwords. Do not create JWT. The frontend only needs a token-shaped value.

### 3.2 Session Routes

Use an in-memory dictionary:

```text
sessions_by_id = {
  session_id: {
    title,
    updated_at,
    messages
  }
}
```

Expose:

- `GET /sessions`
- `GET /sessions/{session_id}`
- `DELETE /sessions/{session_id}`

Return shapes must match the SuperMew frontend expectations.

`GET /sessions` response:

```json
{
  "sessions": [
    {
      "session_id": "session_123",
      "title": "some title",
      "message_count": 2,
      "updated_at": "2026-07-02T18:00:00"
    }
  ]
}
```

`GET /sessions/{session_id}` response:

```json
{
  "messages": [
    {
      "type": "human",
      "content": "hello",
      "timestamp": "2026-07-02T18:00:00",
      "rag_trace": null
    },
    {
      "type": "ai",
      "content": "hi",
      "timestamp": "2026-07-02T18:00:01",
      "rag_trace": null
    }
  ]
}
```

### 3.3 Chat Route

Implement:

`POST /chat/stream`

Request body:

```json
{
  "message": "find all py files",
  "session_id": "session_123"
}
```

Response must be SSE:

```text
data: {"type":"rag_step","step":{"label":"正在理解问题...","icon":"🧠"}}

data: {"type":"rag_step","step":{"label":"正在调用搜索 Agent...","icon":"🔍"}}

data: {"type":"content","content":"partial text"}

data: {"type":"content","content":"more text"}

data: {"type":"trace","rag_trace":{"tool_used":false,"retrieved_chunks":[]}}

data: [DONE]
```

Minimum required event types:

- `content`
- `error`
- `[DONE]`

Optional but recommended:

- `rag_step`
- `trace`
- `session_title`

### 3.4 Document/RAG Stub Routes

Because RAG is reserved for later, implement only empty responses:

- `GET /documents` returns `{ "documents": [] }`
- `POST /documents/upload/async` returns HTTP 501 or a friendly disabled message
- `GET /documents/upload/jobs/{job_id}` returns HTTP 404
- `DELETE /documents/delete/async/{filename}` returns HTTP 501
- `GET /documents/delete/jobs/{job_id}` returns HTTP 404

The preferred UI behavior is to hide the document settings button, so these routes should rarely be used.

## 4. Agent Adapter Scope

Create `backend/agent_adapter.py`.

It must adapt the logic from `langgraph_search_agent.py`, but not import and run its `main()`.

Required behavior:

1. Create the LangGraph ReAct agent once at module startup or through a cached function.
2. Use `ChatOpenAI` with these environment variables:
   - `DEEPSEEK_MODEL`
   - `DEEPSEEK_API_KEY`
   - `DEEPSEEK_BASE_URL`
3. Provide one function for streaming:

```text
stream_search_agent(user_text, history)
```

4. Return chunks/events to `app.py`.
5. Keep per-session message history in `memory_store.py`, not inside the adapter global state.

Important:

The original `find_tool` uses Linux `find`. Replace it with a cross-platform implementation in the new project.

Preferred implementation:

- Use `pathlib.Path(path).rglob(pattern)`.
- Return only files.
- Limit output to 2000 characters.
- Catch invalid paths and permission errors.

Do not use Windows `find.exe`.

## 5. Frontend Scope

Start by copying the SuperMew frontend:

Source:

`E:\AI_agent_Learning\Stage2_学习工具调用、RAG 与记忆\SuperMew\frontend`

Destination:

`E:\AI_agent_Learning\BigDogBarkProject\frontend`

Then simplify these parts:

### 5.1 Keep

Keep:

- `src/components/Chat/*`
- `src/stores/chat.ts`
- `src/stores/auth.ts`
- `src/stores/sessions.ts`
- `src/types/chat.ts`
- `src/types/user.ts`
- `src/utils/api.ts`
- `src/assets/styles/main.css`
- `src/App.vue`
- `src/main.ts`

### 5.2 Hide or Remove

Hide document management in the first version.

Recommended exact change:

- In `src/components/Sidebar.vue`, remove or comment out the settings button.
- In `src/App.vue`, remove the `DocumentSettings` import and the settings branch.
- Do not delete document components unless TypeScript build requires it.

### 5.3 Rename UI Text

Change visible product text:

- `喵喵助手` -> `BigDog Bark`
- `喵喵在线中...` -> `BigDog 在线中...`
- Error text can stay simple Chinese.

Do not redesign the UI in this milestone.

### 5.4 Frontend Contract

Do not change `chat.ts` SSE parser unless absolutely necessary.

The backend should match the existing frontend format:

```text
data: {json}\n\n
```

## 6. Environment Files

Create `.env.example`:

```text
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
HOST=0.0.0.0
PORT=8000
```

Do not commit real API keys.

## 7. Build and Run Commands

Backend:

```powershell
cd E:\AI_agent_Learning\BigDogBarkProject
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install fastapi uvicorn python-dotenv langchain langchain-core langchain-openai langgraph
copy .env.example .env
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```powershell
cd E:\AI_agent_Learning\BigDogBarkProject\frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## 8. Step-by-Step Execution Checklist

### Phase 1: Project Files

1. Confirm `E:\AI_agent_Learning\BigDogBarkProject` exists.
2. Create backend directory and empty `__init__.py`.
3. Copy SuperMew frontend into `BigDogBarkProject\frontend`.
4. Create `.env.example`.
5. Create `README.md` with run instructions.

Stop if any copy command fails.

### Phase 2: Backend Minimal API

1. Create `schemas.py` with request/response models.
2. Create `memory_store.py` with in-memory sessions.
3. Create `rag_stub.py` with empty document routes or helper functions.
4. Create `agent_adapter.py` with:
   - `find_tool`
   - `get_agent`
   - streaming adapter
5. Create `app.py` with:
   - FastAPI app
   - CORS middleware
   - auth routes
   - session routes
   - chat stream route
   - document stub routes
   - static frontend mounting only if `frontend/dist` exists

Stop if Python import fails.

### Phase 3: Frontend Simplification

1. Update visible app name to `BigDog Bark`.
2. Hide settings/document management entry.
3. Remove unused `DocumentSettings` import from `App.vue`.
4. Keep login panel for now, backed by mock auth.
5. Confirm `npm run build` does not report TypeScript errors.

Stop if TypeScript build fails.

### Phase 4: Local Verification

1. Start backend on port 8000.
2. Start frontend on port 3000.
3. Open `http://localhost:3000`.
4. Register or login with any username/password.
5. Send: `请搜索当前目录下的 *.py 文件`.
6. Confirm the bot response streams into the same message bubble.
7. Confirm stop button cancels a running response.
8. Confirm history button opens without crashing.
9. Confirm settings/document UI is not visible.
10. Confirm backend terminal has no uncaught traceback.

## 9. Acceptance Criteria

The implementation is accepted only if all items pass:

- Frontend starts with `npm run dev`.
- Backend starts with `uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload`.
- Mock login works.
- `/chat/stream` returns valid SSE.
- Chat response streams visibly in the frontend.
- No real RAG or document database is required.
- `/documents` returns an empty list or settings UI is hidden.
- Session list and session load do not crash.
- Original SuperMew and original `langgraph_search_agent.py` remain unchanged.

## 10. Common Failure Points and Fixes

### Problem: frontend says login expired

Check:

- `GET /auth/me` exists.
- It returns HTTP 200.
- Response has `username` and `role`.

### Problem: chat sends but no stream appears

Check:

- Backend route is exactly `/chat/stream`.
- Response media type is `text/event-stream`.
- Each event starts with `data: `.
- Each event ends with two newlines.

### Problem: frontend dev server cannot reach backend

Check `frontend/vite.config.ts` has proxies:

```text
/auth -> http://localhost:8000
/chat -> http://localhost:8000
/sessions -> http://localhost:8000
/documents -> http://localhost:8000
```

### Problem: file search fails on Windows

Do not use the Linux `find` command. Use `pathlib.Path.rglob`.

### Problem: build fails because document components are imported

Remove `DocumentSettings` import and settings branch from `App.vue`.

Do not delete random files to make the error disappear.

## 11. Suggested Implementation Order for a Weak Agent

Follow this exact order:

1. Copy frontend.
2. Create backend `app.py` with mock auth only.
3. Run backend and test `/auth/me`.
4. Run frontend and confirm login screen works.
5. Add session routes.
6. Add `/chat/stream` returning hardcoded SSE text.
7. Confirm frontend displays hardcoded streamed text.
8. Only then connect `agent_adapter.py`.
9. Replace hardcoded stream with real agent stream.
10. Hide settings/document UI.
11. Run final verification.

Do not connect the agent before the hardcoded SSE test works.

## 12. Future Work Not Included

These are explicitly out of scope:

- Real RAG retrieval.
- File upload.
- Vector database.
- Persistent user accounts.
- Real JWT auth.
- Multi-user isolation.
- Production deployment.
- UI redesign.
- Docker.

