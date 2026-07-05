# 🐶 BigDogBarkProject — 大狗叫

基于 **LangChain + LangGraph** 的本地文件搜索 Agent 聊天应用，支持对话路由、工具调用（文件搜索/读取/联网搜索）、SSE 流式输出。

---

## 📋 目录

- [架构总览](#架构总览)
- [Agent 设计](#agent-设计)
- [工具调用系统](#工具调用系统)
- [启动器设计与 Harness](#启动器设计与-harness)
- [三套适配器对比](#三套适配器对比)
- [SSE 流式协议](#sse-流式协议)
- [前端架构](#前端架构)
- [SSH 隧道（远端 LM Studio）](#ssh-隧道远端-lm-studio)
- [配置参考](#配置参考)
- [部署与启动](#部署与启动)

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    前端 (Vue 3 + Vite)                       │
│  ChatArea / MessageItem / ThinkingTrace / References        │
│  Pinia stores: chat / auth / sessions / documents           │
└───────────────┬─────────────────────────────────┬───────────┘
                │ HTTP POST /chat/stream           │ npm run dev
                │ SSE event stream                 │ :5173
┌───────────────▼─────────────────────────────────▼───────────┐
│               后端 (FastAPI + Uvicorn)                        │
│  app.py → 路由分发 / auth / sessions / chat/stream           │
│  memory_store.py → 进程内 dict 会话存储                      │
│  schemas.py → Pydantic 请求/响应模型                         │
│  rag_stub.py → RAG 桩模块（预留）                            │
└───────────────┬─────────────────────────────────────────────┘
                │ 动态导入 AGENT_ADAPTER
┌───────────────▼─────────────────────────────────────────────┐
│           Agent 适配器层（三选一）                            │
│                                                              │
│  agent_adapter.py            ← DeepSeek API (LangGraph)     │
│  agent_adapter_local_LLM.py  ← 本地模型 (LangGraph ReAct)   │
│  agent_adapter_local_LLM_harness.py  ← 本地模型 (原生工具循环)│
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│              工具层                                           │
│  find_tool      → os.walk + fnmatch 文件名搜索               │
│  read_file_tool → 读取本地文件内容                           │
│  web_search_tool → DuckDuckGo 联网搜索                       │
└─────────────────────────────────────────────────────────────┘
```

### 启动器（Harness）

```
start.py ───→ 读取 .env ──→ 可选 SSH 隧道 ──→ 启动 uvicorn ──→ 启动 Vite
                 │                 │
             环境变量注入      paramiko SSH 本地转发
```

---

## Agent 设计

### 设计演进

项目经历了两代 Agent 设计：

| 版本 | 适配器 | 架构 | 特点 |
|------|--------|------|------|
| v1 | `agent_adapter.py` | LangGraph `create_react_agent` | DeepSeek API，token 充足 |
| v2 | `agent_adapter_local_LLM.py` | LangGraph ReAct + 路由分类 | 本地 Qwen，先路由再执行 |
| v3 | `agent_adapter_local_LLM_harness.py` | **原生工具循环（Zleap 风格）** | 自制循环，精细控制 tool_call 流量 |

### v3 核心循环（Zleap 风格）

`agent_adapter_local_LLM_harness.py` 实现了不依赖 LangGraph 的**自制 Agent 循环**：

```
        ┌──────────────────────────────────────────┐
        │  1. 构建 System Prompt                    │
        │     - 工具清单 & 调用协议                  │
        │     - 历史上下文摘要                       │
        │     - 预算限制（总调用/连续调用）           │
        └────────────────┬─────────────────────────┘
                         ▼
        ┌──────────────────────────────────────────┐
        │  2. model.ainvoke(messages)              │
        │     ← 模型返回 AIMessage                 │
        └────────────────┬─────────────────────────┘
                         ▼
              ┌──── 有 tool_calls? ────┐
              │                       │
              ▼                       ▼
     ┌─────────────────┐    ┌──────────────────────┐
     │ 3. 执行工具调用   │    │ 4. 生成最终回答       │
     │    for each call │    │    → yield content   │
     │    → 并行执行    │    │    → return          │
     │    → 结果入消息  │    └──────────────────────┘
     │    → 计入预算   │              │
     └────────┬────────┘              │
              │ 还有下一轮             │ 无工具调用
              ▼                       ▼
     ┌──────────────────┐    ┌──────────────────────┐
     │ ← 回到步骤 2     │    │ 5. 缺少工具调用兜底   │
     │   (最多 MAX_TOOL  │    │    → 上下文补发       │
     │    _TURNS 轮)     │    │    → 通用路由器       │
     └──────────────────┘    │    → 连续追问兜底      │
                             └──────────────────────┘
```

### 路由分类

请求首先经过路由判断，决定走"工具"还是"直接回答"：

1. **规则前置**（`_looks_like_tool_request`）：关键词匹配，如"搜索"、"查找"、"联网"、"README"
2. **模型路由**（`_route_query`）：调用小模型二次确认，兜住规则漏掉的边界 case

---

## 工具调用系统

### 注册的工具

| 工具 | 函数 | 说明 |
|------|------|------|
| `find_tool` | `_get_tools()` | 按文件名 glob 在本地目录递归搜索 |
| `read_file_tool` | `_get_tools()` | 读取本地文本文件内容（上限 4000 字符） |
| `web_search_tool` | `_get_tools()` | 通过 DuckDuckGo 联网搜索 |

### 工具调用协议

模型通过结构化 `tool_calls` 发起工具请求，Harness 自行执行而非依赖 LangGraph：

```python
# 模型返回的 tool_call 结构
response.tool_calls = [
    {
        "name": "find_tool",
        "args": {"path": "G:\\project", "pattern": "*.py"},
        "id": "call_xxx"
    }
]

# Harness 自行执行
executed_name, executed_args, content = await _execute_tool_call(call)

# 结果包装为 ToolMessage 追加到消息列表
messages.append(ToolMessage(content=content, tool_call_id=call_id, name=executed_name))
```

### 执行预算控制（关键设计）

本地模型（尤其是小参数模型）容易陷入**工具调用死循环**，Harness 通过三层预算兜底：

| 预算项 | 环境变量 | 默认值 | 行为 |
|--------|----------|--------|------|
| 最大工具回合数 | `MAX_TOOL_TURNS` | 10 | 模型 ↔ 工具 的来回次数 |
| 总工具调用上限 | `MAX_TOTAL_TOOL_CALLS` | 16 | 所有工具调用的累计次数 |
| 连续相同工具上限 | `MAX_CONSECUTIVE_TOOL_CALLS` | 5 | 同一工具不允许无限重试 |

当任意预算耗尽时，Harness 调用 `_build_partial_tool_answer()` 基于已拿到的工具结果给出**阶段性总结**，而非报错退出。

### 缺少工具调用的兜底策略

当模型没有发出 tool_call 但用户意图明显需要工具时，Harness 采用三级兜底：

1. **上下文补发**（`context_followup_tool_call`）：如果用户追问"刚才那个文件"，自动补发 `read_file_tool`
2. **通用工具路由器**（`_route_missing_tool_call`）：调用模型自身做一次路由判断，输出 JSON 指定工具和参数
3. **连续追问**（`continue_nudges`）：给模型一次额外机会"你还没回答用户，请继续"

### 工具结果追踪（Trace）

每个工具调用的结果被记录到 `trace` 字典，最终通过 SSE 发送到前端展示：

```python
trace = {
    "tool_used": bool,
    "tool_name": str,
    "tool_calls": [{"tool_name", "args", "output_preview"}],
    "searched_paths": [{"path", "pattern"}],
    "matched_files": [str],
    "read_files": [str],
    "retrieved_chunks": [{"filename", "text"}],
    "retrieval_stage": str,      # "tool_call_harness"
    "retrieval_mode": str,       # "native_tool_loop"
    "model_name": str,
    "max_model_len": int,
    "max_tool_turns": int,
}
```

---

## 启动器设计与 Harness

### start.py 架构

`start.py` 是项目的**启动编排器**，而非简单的快捷脚本。它的职责链：

```
main()
  ├── ensure_frontend_deps()    → 检查 node_modules（自动 npm install）
  ├── kill_port(PORT)           → 杀掉旧后端进程
  ├── kill_frontend_ports()     → 清理 Vite 旧端口（支持端口扫描范围）
  ├── load_project_env()        → 读取 .env 合并到进程环境
  ├── sanitize_backend_env()    → 清理指向不存在文件的 SSL 证书变量
  ├── maybe_start_llm_tunnel()  → 可选：建立 SSH 本地转发隧道
  ├── subprocess.Popen(uvicorn) → 启动 FastAPI 后端（--reload）
  ├── subprocess.Popen(npm run dev) → 启动 Vite 前端
  └── stop_llm_tunnel()        → 退出时关闭 SSH 隧道
```

### 环境管理

`load_project_env()` 的变量优先级规则：

```
系统环境变量（最高） > .env 文件 > 代码默认值（最低）
```

这意味着：
- 已存在的系统变量不会被 `.env` 覆盖
- 命令行设置优先：`set LLM_MODEL=qwen3.6 && python start.py`

### Windows 进程管理

针对 `uvicorn --reload` 会遗留 worker 进程的问题，实现了递归子进程清理：

```python
# 通过 WMI/CIM 查找子进程树
find_windows_child_pids(PID) → PowerShell Get-CimInstance Win32_Process
kill_windows_process_tree(PID) → taskkill /F /T /PID
```

同时支持扫描端口范围清理 Vite 旧进程（默认 5173-5182）。

### 证书变量清理

Windows Git Bash / Conda 经常设置指向不存在文件的 `SSL_CERT_FILE` 环境变量，导致 httpx/urllib 崩溃。`sanitize_backend_env()` 自动检测并移除这些无效变量。

---

## 三套适配器对比

### agent_adapter.py（DeepSeek API）

```python
# 使用 LangGraph 的 create_react_agent
agent = create_react_agent(model, tools=[find_tool, read_file_tool])
async for update in agent.astream(...):
    # LangGraph 自动管理消息循环
```

- 依赖 `langgraph.prebuilt.create_react_agent`
- 两个工具：`find_tool`, `read_file_tool`
- 无路由分类（默认走工具模式）
- 无需 /no_think 前缀

### agent_adapter_local_LLM.py（本地模型 v1）

- 同样使用 `create_react_agent`
- 三个工具：`find_tool`, `read_file_tool`, `web_search_tool`
- 加入路由分类：`_route_query()` 判断走工具还是直接回答
- 加入 `web_search_tool` 联网搜索
- 解析 `<think>` 标签输出

### agent_adapter_local_LLM_harness.py（本地模型 v2 — 推荐）

最重要的架构升级，关键区别：

| 维度 | v1 (LangGraph ReAct) | v2 (原生工具循环) |
|------|----------------------|-------------------|
| 循环控制 | LangGraph 内部，黑盒 | 自制 `for turn_index in range(MAX_TOOL_TURNS)` |
| 工具绑定 | `create_react_agent` 隐式绑定 | 显式 `model.bind_tools(tools)` |
| 预算控制 | 无（LangGraph 不自带） | 三层预算 + 阶段性总结 |
| 缺少调用兜底 | LangGraph 自动重试 | 三级兜底策略 |
| System Prompt | LangGraph 默认 | 定制化中文 Prompt + 工具清单动态生成 |
| 历史上下文 | 仅 role/content | `assistant_tool_call` + `tool` + `context` 多角色 |
| Debug 能力 | 无内置 | `BIGDOG_LLM_DEBUG=1` 完整打印模型返回对象 |
| 工具结果追踪 | 通过 stream 事件 | 完整 trace + 分段总结 |

### System Prompt 生成

v3 Harness 根据注册的工具列表动态生成 System Prompt：

```python
def _build_tool_inventory_prompt(tools):
    for item in tools:
        name = getattr(item, "name")
        description = getattr(item, "description", "") or item.__doc__
        args = getattr(item, "args", {})
        lines.append(f"- {name}({args.keys()}): {description}")
```

同时注入：
- 工具调用协议（/no_think 模式、上下文优先级）
- 预算限制（总调用数、连续调用数）
- 历史上下文摘要（`history_context_summary`）

---

## SSE 流式协议

前端 ↔ 后端通过 **Server-Sent Events** 通信。

### 请求

```http
POST /chat/stream
Content-Type: application/json

{
    "message": "查找所有 Python 文件",
    "session_id": "session_1749123456789"
}
```

### 响应流

```
data: {"type": "rag_step", "step": {"label": "模型回合 1", "icon": "🧠"}}

data: {"type": "rag_step", "step": {"label": "调用工具：find_tool", "icon": "🔎", "detail": "{'path': '.', 'pattern': '*.py'}"}}

data: {"type": "rag_step", "step": {"label": "find_tool 返回结果", "icon": "📄", "detail": "..."}}

data: {"type": "trace", "rag_trace": {"tool_used": true, ...}}

data: {"type": "thinking", "content": "...推理过程..."}

data: {"type": "content", "content": "最终回答..."}

data: [DONE]
```

### 事件类型

| type | 用途 | 前端处理 |
|------|------|----------|
| `rag_step` | 过程步骤（工具调用进度） | `ThinkingTrace.vue` 展示步骤列表 |
| `trace` | 完整 RAG 轨迹 | `RetrievalTraceDetails.vue` 展示检索详情 |
| `thinking` | 模型推理过程 | 折叠展示在消息中 |
| `content` | 最终回答文本 | `MessageContent.vue` Markdown 渲染 |
| `error` | 错误信息 | 红色文字提示 |
| `[DONE]` | 流结束标记 | 关闭连接 |

---

## 前端架构

### 组件树

```
App.vue
├── Sidebar.vue              ← 左侧导航（新对话/历史记录）
├── HistorySidebar.vue       ← 历史会话列表（滑入）
└── ChatArea.vue             ← 主聊天区域
    ├── WelcomeScreen.vue    ← 空状态欢迎页
    ├── MessageItem.vue      ← 每条消息
    │   ├── MessageContent.vue    ← Markdown 渲染（含引用链接）
    │   ├── ThinkingTrace.vue     ← 思考/推理过程
    │   ├── References.vue        ← 参考文献列表
    │   └── RetrievalTraceDetails.vue ← 检索过程详情
    └── ChatInput.vue        ← 输入框（Shift+Enter 换行）
```

### 流式渲染

`chat store` 的 `handleSend()` 方法通过 `fetch + ReadableStream` 消费 SSE：

```typescript
// 流式 SSE 消费（stores/chat.ts）
const reader = response.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    // 解析 data: 事件
    // 按 type 分发到消息状态:
    //   content → 追加文本
    //   rag_step → 更新步骤列表
    //   trace → 存储检索轨迹
}
```

关键特性：
- **自动滚动**：仅当用户已在底部时跟随新内容
- **引用点击**：`[1]` 标记转为可点击引用，点击跳转到对应参考文献
- **分组折叠**：子 Agent 的步骤按组折叠/展开

---

## SSH 隧道（远端 LM Studio）

当本地没有 GPU，但远端有一台运行 LM Studio 的机器时，`start.py` 能自动建立 SSH 本地转发。

### 架构

```
┌────────────────────┐          SSH          ┌──────────────────┐
│ Windows 本地       │ ──────────────────→   │ 远端 Ubuntu      │
│ 127.0.0.1:51234   │    local → remote     │ 127.0.0.1:1234   │
│                    │   端口转发              │ (LM Studio)      │
└────────┬───────────┘                       └──────────────────┘
         │
   LLM_BASE_URL=http://127.0.0.1:51234/v1
         │
    Agent 通过此地址调用模型
```

### 配置 .env

```ini
LLM_SSH_TUNNEL=1
LLM_REMOTE_HOST=your.server.com
LLM_REMOTE_USER=your_username
LLM_REMOTE_PASSWORD=your_password
LLM_REMOTE_PORT=1234
LLM_LOCAL_PORT=51234
```

### 实现细节

```python
# start.py 使用 paramiko 实现端口转发
client = paramiko.SSHClient()
client.connect(remote_host, username=remote_user, password=remote_password)

# 本地 TCP Server → SSH Channel → 远端 LM Studio
server = _ForwardServer(
    (local_host, local_port),
    _make_forward_handler(client.get_transport(), remote_bind_host, remote_port)
)
```

- 启动后自动检测 `http://127.0.0.1:51234/v1/models` 是否可达
- 若本地已有 LLM 服务（如 Qwen LM Studio），跳过隧道建立
- `--no-llm-tunnel` 可强制禁用
- 退出时自动关闭隧道

---

## 配置参考

### 完整 .env

```ini
# ─── Agent 适配器 ───
AGENT_ADAPTER=agent_adapter_local_LLM_harness
# 可选: agent_adapter | agent_adapter_local_LLM | agent_adapter_local_LLM_harness

# ─── 本地 LLM ───
LLM_MODEL=qwen3-4b
LLM_BASE_URL=http://127.0.0.1:51234/v1
LLM_API_KEY=not-needed
LLM_MAX_TOKENS=2048
LLM_NO_THINK=1
MAX_MODEL_LEN=32768

# ─── 工具调用预算 ───
MAX_TOOL_TURNS=10
MAX_TOTAL_TOOL_CALLS=16
MAX_CONSECUTIVE_TOOL_CALLS=5

# ─── SSH 隧道（远端 LM Studio） ───
LLM_SSH_TUNNEL=1
LLM_REMOTE_HOST=your.server.com
LLM_REMOTE_USER=your_username
LLM_REMOTE_PASSWORD=your_password
LLM_REMOTE_PORT=1234
LLM_LOCAL_PORT=51234

# ─── 服务端口 ───
HOST=0.0.0.0
PORT=8000
```

### 环境变量速查

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AGENT_ADAPTER` | `agent_adapter` | 选择适配器 |
| `LLM_MODEL` | `qwen3-4b` | 模型名称 |
| `LLM_BASE_URL` | `http://127.0.0.1:51234/v1` | OpenAI 兼容 API 地址 |
| `LLM_API_KEY` | `not-needed` | API Key |
| `LLM_MAX_TOKENS` | `2048` | 最大输出 token |
| `LLM_NO_THINK` | `1` | 是否添加 /no_think 前缀 |
| `MAX_MODEL_LEN` | `32768` | 模型上下文预算 |
| `MAX_TOOL_TURNS` | `10` | 最大工具回合数 |
| `MAX_TOTAL_TOOL_CALLS` | `16` | 总工具调用上限 |
| `MAX_CONSECUTIVE_TOOL_CALLS` | `5` | 连续相同工具上限 |
| `LLM_SSH_TUNNEL` | `0` | 启用 SSH 隧道 |
| `BIGDOG_LLM_DEBUG` | `0` | 调试模式：打印模型完整返回 |
| `PORT` | `8000` | 后端端口 |
| `HOST` | `0.0.0.0` | 监听地址 |

---

## 部署与启动

### 前置要求

- Python ≥ 3.11（`asyncio.timeout()` 需要）
- Node.js ≥ 18
- conda 或 venv

### 快速安装

```bash
# 方式一：一键脚本
bash setup.sh

# 方式二：手动
conda create -n bigdog python=3.11 -y
conda activate bigdog
pip install "fastapi[standard]" uvicorn langchain langchain-openai langgraph python-dotenv
cd frontend && npm install
```

### 启动

```bash
# DeepSeek API
python start.py

# 本地模型（Qwen / LM Studio）
python start.py --adapter agent_adapter_local_LLM_harness

# 调试模式（打印模型返回对象）
python start.py --adapter agent_adapter_local_LLM_harness --debug-llm

# 禁用 SSH 隧道
python start.py --no-llm-tunnel
```

### 单独启停

```bash
# 仅后端
conda activate bigdog
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 仅前端
cd frontend && npm run dev
```

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 前端 | Vue 3 + TypeScript | UI 框架 |
| 前端 | Pinia | 状态管理 |
| 前端 | Vite | 构建工具 |
| 前端 | marked + highlight.js | Markdown 渲染 + 代码高亮 |
| 后端 | FastAPI + Uvicorn | Web 框架 |
| 后端 | LangChain + LangGraph | Agent 框架（v1/v2） |
| 后端 | httpx / urllib | HTTP 请求 |
| 后端 | paramiko | SSH 隧道（可选） |
| 模型 | OpenAI 兼容 API | DeepSeek / Qwen / LM Studio |
| 搜索 | DuckDuckGo Search (`ddgs`) | 联网搜索 |
| 启动器 | subprocess + socketserver | 进程编排 + 端口转发 |
