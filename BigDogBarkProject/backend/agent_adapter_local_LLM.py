"""LangGraph search agent adapter for BigDogBarkProject.

The stream is deliberately split into two phases:
1. run the tool agent and expose only process events / trace data;
2. generate a clean final answer from the collected tool context.
"""

import asyncio
import fnmatch
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
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
MAX_VISITED_FILES = 20000
MAX_MATCHES = 80
MAX_RETURN_CHARS = 2000
MAX_READ_CHARS = 4000
AGENT_TIMEOUT_SECONDS = 90
FINAL_ANSWER_TIMEOUT_SECONDS = 60

_agent = None
_model = None


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
    """Recursively search for files by filename pattern."""
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
    """Read a text file and return a bounded excerpt."""
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


def get_model() -> ChatOpenAI:
    """Return the shared chat model."""
    global _model
    if _model is None:
        _model = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "qwen3.6-27b-awq-int4"),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
            base_url=os.getenv("LLM_BASE_URL", "http://100.81.149.79:8000/v1"),
            temperature=0,
        )
    return _model


def get_agent():
    """Return the shared tool agent."""
    global _agent
    if _agent is None:
        _agent = create_react_agent(get_model(), tools=[find_tool, read_file_tool])
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
            f"[{index}] tool={item['tool_name']} args={item['args']}\n{item['content']}"
        )
    context = "\n\n".join(context_lines) or "No tool context was collected."

    system = (
        "You are BigDog, a concise assistant. Produce only the final answer for the user. "
        "Do not narrate internal search steps, tool calls, failed patterns, or raw logs. "
        "If files were found, summarize the useful results and mention relevant filenames. "
        "If the context is insufficient, say what is missing briefly."
    )
    user = (
        f"User question:\n{user_text}\n\n"
        f"Collected search context:\n{context}\n\n"
        f"Trace summary:\n"
        f"- searched_paths: {trace.get('searched_paths', [])}\n"
        f"- matched_files: {trace.get('matched_files', [])}\n"
        "Now write the clean final answer in Chinese."
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]


async def stream_search_agent(user_text: str, history: list[dict]):
    """Yield SSE event dictionaries for process trace and final answer."""
    agent = get_agent()
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

    yield {"type": "rag_step", "step": {"label": "正在理解问题...", "icon": "🧠"}}

    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            async for update in agent.astream(
                {"messages": _build_agent_messages(user_text, history)},
                stream_mode="updates",
                config={"recursion_limit": 8},
            ):
                for node_name, output in update.items():
                    if not isinstance(output, dict) or "messages" not in output:
                        continue
                    msg = output["messages"][-1]

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
            async for chunk in get_model().astream(_build_final_prompt(user_text, trace, tool_context)):
                if not isinstance(chunk, AIMessageChunk):
                    continue

                # 分离推理过程（本地 Qwen 的 reasoning_content 字段）
                reasoning = getattr(chunk, "reasoning_content", None) or ""
                if reasoning:
                    yield {"type": "thinking", "content": reasoning}

                # 普通文本内容
                content = _message_content_to_text(chunk.content)
                if content:
                    yield {"type": "content", "content": content}
    except asyncio.TimeoutError:
        yield {"type": "error", "content": f"最终回答生成超时：超过 {FINAL_ANSWER_TIMEOUT_SECONDS}s"}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield {"type": "error", "content": f"最终回答生成出错: {str(e)}"}
