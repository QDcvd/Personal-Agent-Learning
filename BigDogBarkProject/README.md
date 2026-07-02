# 🐶 BigDogBarkProject

基于 LangGraph ReAct 搜索 Agent 的聊天应用。

## 目录结构

```
BigDogBarkProject/
├── backend/
│   ├── __init__.py
│   ├── app.py              # FastAPI 主应用
│   ├── schemas.py          # Pydantic 模型
│   ├── memory_store.py     # 内存会话存储
│   ├── rag_stub.py         # RAG 桩（预留）
│   └── agent_adapter.py    # LangGraph Agent 适配器
├── frontend/               # Vue3 + Vite 前端
├── .env.example
├── README.md
└── IMPLEMENTATION_PLAN.md
```

## 启动

### 后端

```powershell
cd BigDogBarkProject
pip install fastapi uvicorn python-dotenv langchain langchain-core langchain-openai langgraph
copy .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 前端

```powershell
cd BigDogBarkProject/frontend
npm install
npm run dev
```

打开 http://localhost:3000
