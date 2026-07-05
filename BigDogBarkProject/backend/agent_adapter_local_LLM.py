"""LangGraph search agent adapter for BigDogBarkProject.

The stream is deliberately split into two phases:
1. run the tool agent and expose only process events / trace data;
2. generate a clean final answer from the collected tool context.
"""

import asyncio
import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

try:
    from langchain.agents import create_react_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent

load_dotenv()


EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
}
MAX_SEARCH_SECONDS = 5.0
MAX_VISITED_FILES = 90000
MAX_MATCHES = 80
MAX_RETURN_CHARS = 2000
MAX_READ_CHARS = 4000
AGENT_TIMEOUT_SECONDS = 900
FINAL_ANSWER_TIMEOUT_SECONDS = 180
ROUTE_TIMEOUT_SECONDS = 8

FINAL_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?(?:final\s+output\s+generation|final\s+answer|"
    r"final\s+response|final|answer|response|最终回答|最终答案|最终|答案|回答)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
THINKING_MARKER_PATTERN = re.compile(
    r"(?:here(?:'s| is)\s+a\s+thinking\s+process|thinking\s+process|reasoning\s+process|"
    r"analysis\s+process|internal\s+reasoning|analyze\s+user\s+input|check\s+constraints|"
    r"identify\s+key\s+constraints|思考过程|推理过程|分析过程)\s*[:：]",
    flags=re.IGNORECASE,
)
_agent = None
_model = None


def _llm_debug_enabled() -> bool:
    return os.getenv("BIGDOG_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes", "on", "debug"}


def _debug_plain_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _debug_plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_plain_value(item) for item in value]
    return value


def _debug_dump_model_return(label: str, value: Any) -> None:
    if not _llm_debug_enabled():
        return

    payload = {
        "标签": label,
        "类型": f"{type(value).__module__}.{type(value).__name__}",
        "完整对象": _debug_plain_value(value),
        "repr": repr(value),
    }
    if isinstance(value, (AIMessage, AIMessageChunk)):
        payload["content"] = getattr(value, "content", None)
        payload["tool_calls"] = getattr(value, "tool_calls", None)
        payload["invalid_tool_calls"] = getattr(value, "invalid_tool_calls", None)
        payload["additional_kwargs"] = getattr(value, "additional_kwargs", None)
        payload["response_metadata"] = getattr(value, "response_metadata", None)
        payload["usage_metadata"] = getattr(value, "usage_metadata", None)
        payload["id"] = getattr(value, "id", None)

    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = repr(payload)

    print("\n========== 大模型返回值调试开始 ==========", file=sys.stderr, flush=True)
    print(text, file=sys.stderr, flush=True)
    print("========== 大模型返回值调试结束 ==========\n", file=sys.stderr, flush=True)


def _debug_log(message: str) -> None:
    if _llm_debug_enabled():
        print(f"[llm-debug] {message}", file=sys.stderr, flush=True)


def _format_limit_notes(
    skipped_dirs: int,
    timed_out: bool,
    hit_file_limit: bool,
    hit_match_limit: bool,
) -> list[str]:
    notes = []
    if skipped_dirs:
        notes.append(f"Skipped {skipped_dirs} dependency/cache/hidden directories.")
    if timed_out:
        notes.append(f"Stopped after {MAX_SEARCH_SECONDS:.0f}s search limit.")
    if hit_file_limit:
        notes.append(f"Stopped after visiting {MAX_VISITED_FILES} files.")
    if hit_match_limit:
        notes.append(f"Stopped after {MAX_MATCHES} matches.")
    return notes


def _extract_paths_from_tool_text(text: str) -> list[str]:
    paths = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("...") or stripped.startswith("Skipped "):
            continue
        if stripped.startswith("Stopped ") or stripped.startswith("No files matched"):
            continue
        if "\\" in stripped or "/" in stripped:
            paths.append(stripped)
    return paths


@tool
def find_tool(path: str = ".", pattern: str = "*") -> str:
    """按文件名模式递归查找本地文件。"""
    try:
        root = Path(path).expanduser()
        if not root.exists():
            return f"Directory does not exist: {path}"
        if not root.is_dir():
            return f"Path is not a directory: {path}"

        deadline = time.monotonic() + MAX_SEARCH_SECONDS
        visited = 0
        skipped_dirs = 0
        timed_out = False
        hit_file_limit = False
        files: list[Path] = []

        for current_root, dirs, filenames in os.walk(root, topdown=True):
            original_dir_count = len(dirs)
            dirs[:] = [
                dirname
                for dirname in dirs
                if dirname not in EXCLUDED_DIRS and not dirname.startswith(".")
            ]
            skipped_dirs += original_dir_count - len(dirs)

            for filename in filenames:
                if time.monotonic() > deadline:
                    timed_out = True
                    break

                visited += 1
                if visited > MAX_VISITED_FILES:
                    hit_file_limit = True
                    break

                if fnmatch.fnmatch(filename, pattern):
                    files.append(Path(current_root) / filename)
                    if len(files) >= MAX_MATCHES:
                        break

            if timed_out or hit_file_limit or len(files) >= MAX_MATCHES:
                break

        lines = [str(p) for p in files[:50]]
        result = "\n".join(lines)
        if not result:
            result = f"No files matched {pattern} under {path}."
        if len(result) > MAX_RETURN_CHARS:
            result = result[:MAX_RETURN_CHARS] + "\n...(truncated)"
        if len(files) > 50:
            result += f"\n... {len(files)} total matches, showing first 50."

        notes = _format_limit_notes(
            skipped_dirs=skipped_dirs,
            timed_out=timed_out,
            hit_file_limit=hit_file_limit,
            hit_match_limit=len(files) >= MAX_MATCHES,
        )
        if notes:
            result += "\n" + "\n".join(notes)
        return result
    except PermissionError as e:
        return f"Permission denied: {e}"
    except Exception as e:
        return f"Search failed: {e}"


@tool
def read_file_tool(path: str) -> str:
    """读取本地文本文件，并返回长度受限的内容片段。"""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"File does not exist: {path}"
        if not p.is_file():
            return f"Path is not a file: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_READ_CHARS:
            content = content[:MAX_READ_CHARS] + "\n...(truncated)"
        return content
    except Exception as e:
        return f"Read failed: {e}"


@tool
def web_search_tool(query: str) -> str:
    """在互联网上搜索信息"""
    import subprocess
    result = subprocess.run(
        ["curl", "-s", f"https://html.duckduckgo.com/html/?q={query}"],
        capture_output=True, text=True, timeout=15
    )
    return result.stdout[:2000]


def _get_tools() -> list[Any]:
    """返回暴露给 Agent 的工具列表。"""
    return [find_tool, read_file_tool, web_search_tool]


def _build_tool_inventory_prompt(tools: list[Any]) -> str:
    """根据实际注册工具生成中文系统提示词片段。"""
    lines = [
        "可用工具列表：",
        "当用户要求搜索、读取文件、查看项目，或明确点名某个工具时，你可以使用下列工具。",
    ]
    for item in tools:
        name = getattr(item, "name", getattr(item, "__name__", "tool"))
        description = (getattr(item, "description", None) or getattr(item, "__doc__", "") or "").strip()
        args = getattr(item, "args", None)
        if isinstance(args, dict) and args:
            args_text = ", ".join(args.keys())
        else:
            args_text = "见工具参数结构"
        lines.append(f"- {name}({args_text}): {description}")
    return "\n".join(lines)


def get_model() -> ChatOpenAI:
    """Return the shared chat model."""
    global _model
    if _model is None:
        _model = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "qwen3.6"),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            temperature=0,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        )
    return _model


def get_agent():
    """Return the shared tool agent."""
    global _agent
    if _agent is None:
        tools = _get_tools()
        agent_prompt = (
            "你是 BigDog，一个会使用工具的中文助手。\n"
            f"{_build_tool_inventory_prompt(tools)}\n\n"
            "工具使用规则：\n"
            "- 如果用户明确点名某个工具，就调用该工具。\n"
            "- 如果用户要求网页搜索、网络搜索、联网查询、在线查找、最新信息，或说“搜索”“网络搜索”“联网查询”，就调用 web_search_tool。\n"
            "- 如果用户要求查找本地项目文件，就调用 find_tool。\n"
            "- 如果用户要求查看、阅读、总结某个具体本地文件，就调用 read_file_tool。\n"
            "- 只要可用工具能够满足用户请求，就不要声称自己无法使用工具。"
        )
        _agent = create_react_agent(get_model(), tools=tools, prompt=agent_prompt)
    return _agent


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, str):
                text += block
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        return text
    return ""


def _message_reasoning_to_text(chunk: AIMessageChunk) -> str:
    reasoning = getattr(chunk, "reasoning_content", None)
    if isinstance(reasoning, str):
        return reasoning

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
    reasoning = additional_kwargs.get("reasoning_content")
    if isinstance(reasoning, str):
        return reasoning

    content = getattr(chunk, "content", None)
    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, dict):
                value = block.get("reasoning_content") or block.get("reasoning")
                if isinstance(value, str):
                    text += value
        return text
    return ""


def _pick_final_answer(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    chinese_lines = [line for line in lines if re.search(r"[\u4e00-\u9fff]", line)]
    if chinese_lines:
        return chinese_lines[-1].strip("“”\"' ")
    return text.strip()


def _split_local_llm_output(text: str) -> tuple[str, str]:
    """Return (thinking, answer) while keeping Qwen analysis out of final content."""
    if not text:
        return "", ""

    thinking_parts = [
        block.strip()
        for block in re.findall(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
        if block.strip()
    ]
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    final_matches = list(FINAL_MARKER_PATTERN.finditer(cleaned))
    if final_matches:
        marker = final_matches[-1]
        prefix = cleaned[:marker.start()].strip()
        answer_source = cleaned[marker.end():].strip()
        if prefix:
            thinking_parts.append(prefix)
        answer = _pick_final_answer(answer_source)
        trailing_thinking = answer_source[: answer_source.rfind(answer)].strip() if answer else ""
        if trailing_thinking:
            thinking_parts.append(trailing_thinking)
        return "\n\n".join(thinking_parts).strip(), answer

    if THINKING_MARKER_PATTERN.search(cleaned):
        answer = _pick_final_answer(cleaned)
        thinking = cleaned[: cleaned.rfind(answer)].strip() if answer else cleaned
        if thinking:
            thinking_parts.append(thinking)
        return "\n\n".join(thinking_parts).strip(), answer

    return "\n\n".join(thinking_parts).strip(), cleaned


def _tool_call_name(call: Any) -> str:
    if isinstance(call, dict):
        return call.get("name") or call.get("function", {}).get("name") or "tool"
    return getattr(call, "name", "tool")


def _tool_call_args(call: Any) -> dict:
    if isinstance(call, dict):
        args = call.get("args") or call.get("arguments") or {}
        return args if isinstance(args, dict) else {"raw": str(args)}
    args = getattr(call, "args", {})
    return args if isinstance(args, dict) else {"raw": str(args)}


def _append_tool_result_to_trace(trace: dict, tool_name: str, args: dict, content: str) -> None:
    trace["tool_used"] = True
    trace["tool_name"] = tool_name
    trace["tool_calls"].append(
        {
            "tool_name": tool_name,
            "args": args,
            "output_preview": content[:800],
        }
    )

    if tool_name == "find_tool":
        path = str(args.get("path", "."))
        pattern = str(args.get("pattern", "*"))
        trace["searched_paths"].append({"path": path, "pattern": pattern})
        for matched_path in _extract_paths_from_tool_text(content):
            if matched_path not in trace["matched_files"]:
                trace["matched_files"].append(matched_path)
                trace["retrieved_chunks"].append(
                    {
                        "filename": matched_path,
                        "text": "Matched by local file search.",
                    }
                )
    elif tool_name == "read_file_tool":
        filename = str(args.get("path", ""))
        trace["read_files"].append(filename)
        trace["retrieved_chunks"].append(
            {
                "filename": filename,
                "text": content[:1000],
            }
        )


def _build_agent_messages(user_text: str, history: list[dict]) -> list[Any]:
    messages: list[Any] = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(("user", content))
        elif role == "assistant":
            messages.append(("ai", content))
    messages.append(("user", user_text))
    return messages


def _build_final_prompt(user_text: str, trace: dict, tool_context: list[dict]) -> list[Any]:
    context_lines = []
    for index, item in enumerate(tool_context, start=1):
        context_lines.append(
            f"[{index}] 工具={item['tool_name']} 参数={item['args']}\n{item['content']}"
        )
    context = "\n\n".join(context_lines) or "没有收集到工具上下文。"

    system = (
        "你是 BigDog，一个简洁的中文助手。请只输出给用户看的最终回答。"
        "不要复述内部搜索步骤、工具调用过程、失败模式或原始日志。"
        "如果找到了文件或网页信息，请总结有用结果，并在必要时提到相关文件名或来源。"
        "如果上下文不足，请简短说明还缺少什么。"
    )
    user = (
        f"用户问题：\n{user_text}\n\n"
        f"已收集的工具上下文：\n{context}\n\n"
        f"检索摘要：\n"
        f"- 搜索路径：{trace.get('searched_paths', [])}\n"
        f"- 匹配文件：{trace.get('matched_files', [])}\n"
        "请用中文写出干净、直接的最终回答。"
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _build_route_prompt(user_text: str) -> list[Any]:
    return [
        SystemMessage(
            content=(
                "你是请求路由器，只能输出一个英文单词：tools 或 direct_chat。\n"
                "可用工具名单：find_tool, read_file_tool, web_search_tool。\n"
                "输出 tools：用户要求搜索、查找、读取文件、查看项目、README、代码、路径、联网搜索、网络搜索、搜索一下、最新信息，或明确提到任一工具名。\n"
                "输出 direct_chat：普通闲聊、身份问题、解释概念、写作润色，且不需要任何工具。\n"
                "示例：使用web_search_tool进行搜索 => tools\n"
                "示例：网络搜索大狗大狗叫叫叫是什么东西 => tools\n"
                "示例：查找 README => tools\n"
                "示例：你是什么模型 => direct_chat\n"
                "不要解释，不要输出标点。"
            )
        ),
        HumanMessage(content=f"用户请求：\n{user_text}\n\n路由结果："),
    ]


async def _route_query(user_text: str) -> str:
    try:
        async with asyncio.timeout(ROUTE_TIMEOUT_SECONDS):
            _debug_log("旧local即将请求路由模型")
            result = await get_model().ainvoke(_build_route_prompt(user_text))
            _debug_dump_model_return("旧local路由模型返回", result)
    except Exception:
        return "direct_chat"

    raw = _message_reasoning_to_text(result) + _message_content_to_text(result.content)
    answer = _split_local_llm_output(raw)[1].lower()
    return "tools" if re.search(r"\btools\b", answer) else "direct_chat"


def _build_direct_chat_prompt(user_text: str, history: list[dict]) -> list[Any]:
    messages: list[Any] = [
        SystemMessage(
            content=(
                "你是 BigDog，一个简洁的中文助手。请直接回答用户。"
                "不要提及工具调用、内部分析或隐藏推理。"
            )
        )
    ]
    for msg in history[-6:]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(("user", content))
        elif role == "assistant":
            messages.append(("ai", content))
    messages.append(HumanMessage(content=user_text))
    return messages


async def _stream_model_answer(messages: list[Any]):
    raw_output = ""
    _debug_log(f"旧local即将开始流式请求大模型，消息数量={len(messages)}")
    async for chunk in get_model().astream(messages):
        _debug_dump_model_return("旧local流式模型返回片段", chunk)
        if not isinstance(chunk, AIMessageChunk):
            continue

        raw_output += _message_reasoning_to_text(chunk)
        raw_output += _message_content_to_text(chunk.content)

    thinking, answer = _split_local_llm_output(raw_output)
    if thinking:
        yield {"type": "thinking", "content": thinking}
    if answer:
        yield {"type": "content", "content": answer}


async def stream_search_agent(user_text: str, history: list[dict]):
    """Yield SSE event dictionaries for process trace and final answer."""
    trace = {
        "tool_used": False,
        "tool_name": None,
        "tool_calls": [],
        "searched_paths": [],
        "matched_files": [],
        "read_files": [],
        "retrieved_chunks": [],
        "retrieval_stage": "local_tool_search",
        "retrieval_mode": "agent_tools",
    }
    tool_context: list[dict] = []
    pending_tool_args: dict[str, dict] = {}

    route = await _route_query(user_text)
    if route == "direct_chat":
        trace["retrieval_stage"] = "direct_chat"
        trace["retrieval_mode"] = "no_tools"
        yield {"type": "trace", "rag_trace": trace}
        yield {"type": "rag_step", "step": {"label": "正在直接回应...", "icon": "✍️"}}
        try:
            async with asyncio.timeout(FINAL_ANSWER_TIMEOUT_SECONDS):
                async for event in _stream_model_answer(_build_direct_chat_prompt(user_text, history)):
                    yield event
        except asyncio.TimeoutError:
            yield {"type": "error", "content": f"最终回答生成超时：超过 {FINAL_ANSWER_TIMEOUT_SECONDS}s"}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield {"type": "error", "content": f"最终回答生成出错: {str(e)}"}
        return

    agent = get_agent()

    yield {"type": "rag_step", "step": {"label": "正在理解问题...", "icon": "🧠"}}

    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            _debug_log("旧local即将开始 agent.astream 工具流程")
            async for update in agent.astream(
                {"messages": _build_agent_messages(user_text, history)},
                stream_mode="updates",
                config={"recursion_limit": 8},
            ):
                for node_name, output in update.items():
                    if not isinstance(output, dict) or "messages" not in output:
                        continue
                    msg = output["messages"][-1]
                    _debug_dump_model_return(f"旧local agent节点返回-{node_name}", msg)

                    tool_calls = getattr(msg, "tool_calls", None) or []
                    for call in tool_calls:
                        tool_name = _tool_call_name(call)
                        args = _tool_call_args(call)
                        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
                        if call_id:
                            pending_tool_args[call_id] = {"tool_name": tool_name, "args": args}
                        yield {
                            "type": "rag_step",
                            "step": {
                                "label": f"调用工具：{tool_name}",
                                "icon": "🔎",
                                "detail": str(args),
                            },
                        }

                    if node_name != "tools":
                        continue

                    content = _message_content_to_text(getattr(msg, "content", ""))
                    tool_call_id = getattr(msg, "tool_call_id", "")
                    pending = pending_tool_args.get(tool_call_id, {})
                    tool_name = getattr(msg, "name", "") or pending.get("tool_name", "tool")
                    args = pending.get("args", {})

                    _append_tool_result_to_trace(trace, tool_name, args, content)
                    tool_context.append(
                        {
                            "tool_name": tool_name,
                            "args": args,
                            "content": content,
                        }
                    )
                    yield {
                        "type": "rag_step",
                        "step": {
                            "label": f"{tool_name} 返回结果",
                            "icon": "📄",
                            "detail": content[:180],
                        },
                    }
    except asyncio.TimeoutError:
        yield {"type": "error", "content": f"搜索超时：超过 {AGENT_TIMEOUT_SECONDS}s，已停止本轮工具搜索"}
        yield {"type": "trace", "rag_trace": trace}
        return
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield {"type": "error", "content": f"搜索出错: {str(e)}"}
        yield {"type": "trace", "rag_trace": trace}
        return

    yield {"type": "trace", "rag_trace": trace}
    yield {"type": "rag_step", "step": {"label": "正在整理最终回答...", "icon": "✍️"}}

    try:
        async with asyncio.timeout(FINAL_ANSWER_TIMEOUT_SECONDS):
            async for event in _stream_model_answer(_build_final_prompt(user_text, trace, tool_context)):
                yield event
    except asyncio.TimeoutError:
        yield {"type": "error", "content": f"最终回答生成超时：超过 {FINAL_ANSWER_TIMEOUT_SECONDS}s"}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield {"type": "error", "content": f"最终回答生成出错: {str(e)}"}
