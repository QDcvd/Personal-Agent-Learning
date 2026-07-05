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
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from backend.context_manager import (
    context_followup_tool_call,
    history_context_summary,
    recent_context_paths,
)

load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
MAX_MATCHES = 800
MAX_RETURN_CHARS = 2000
MAX_READ_CHARS = 4000
AGENT_TIMEOUT_SECONDS = 900
FINAL_ANSWER_TIMEOUT_SECONDS = 180
DEFAULT_LLM_MODEL = "qwen3-4b"
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 2048)
LLM_NO_THINK = _env_bool("LLM_NO_THINK", True)
MAX_MODEL_LEN = _env_int("MAX_MODEL_LEN", _env_int("LLM_MAX_MODEL_LEN", 32768))
MAX_TOOL_TURNS = _env_int("MAX_TOOL_TURNS", 10)
MAX_TOTAL_TOOL_CALLS = _env_int("MAX_TOTAL_TOOL_CALLS", 16)
MAX_CONSECUTIVE_TOOL_CALLS = _env_int("MAX_CONSECUTIVE_TOOL_CALLS", 5)

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
    """调试模式下打印大模型返回的完整对象，便于排查 tool_call 解析问题。"""
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
    """按文件名通配符在真实本地目录中查找文件；path 必须是存在目录如 G:\\ 或 G:\\project，pattern 必须是文件名 glob 如 *.md、package.json、*blog*，不要传自然语言。"""
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
    """读取一个真实存在的本地文本文件；path 必须是完整文件路径，不要传目录或自然语言问题。"""
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
    """联网搜索公开网页信息；query 应是搜索关键词或问题，不要传本地路径或文件名通配符。"""
    try:
        from ddgs import DDGS
    except ImportError:
        return "网络搜索失败：缺少 ddgs 包。请运行 pip install ddgs。"

    try:
        results = list(DDGS(timeout=15).text(query, max_results=5))
    except Exception as e:
        return f"网络搜索失败：{type(e).__name__}: {e}"[:MAX_RETURN_CHARS]

    clean_results = []
    for item in results:
        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        body = str(item.get("body") or item.get("snippet") or "").strip()
        if title or url or body:
            clean_results.append((title, url, body))

    if not clean_results:
        return "网络搜索没有返回内容。"

    lines = [f"搜索结果：{query}"]
    for index, (title, url, body) in enumerate(clean_results, start=1):
        lines.append(f"{index}. {title or '无标题'}")
        if url:
            lines.append(f"链接：{url}")
        if body:
            lines.append(f"摘要：{body}")
    return "\n".join(lines)[:MAX_RETURN_CHARS]


def _get_tools() -> list[Any]:
    """返回暴露给 Agent 的工具列表。"""
    return [find_tool, read_file_tool, web_search_tool]


def _build_tool_inventory_prompt(tools: list[Any]) -> str:
    """根据实际注册工具生成中文系统提示词片段。"""
    lines = [
        "可用工具列表：",
        "下面的工具列表来自运行时实际注册的 tools 参数。只要用户请求需要外部信息、项目文件、文件内容，或明确点名某个工具，就优先使用对应工具。",
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
            model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:51234/v1"),
            temperature=0,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _model


def _build_tool_loop_system_prompt(tools: list[Any]) -> str:
    return (
        "你是 BigDog，一个会使用工具的中文助手。\n"
        f"{_build_tool_inventory_prompt(tools)}\n\n"
        "工具调用协议：\n"
        "- 运行时已经把上面的工具作为结构化 tools schema 传给模型；需要工具时必须发起正式 tool_call，不要在文字里假装调用。\n"
        "- 当前使用 /no_think 模式，请不要输出隐藏推理过程，把 token 留给工具调用和最终答案。\n"
        "- 如果用户明确点名某个工具，就调用该工具。\n"
        "- 如果用户要求网页搜索、网络搜索、联网查询、在线查找、最新信息，或说“搜一下/搜下/网络搜索/联网查询”，或者你收到你不确定的询问，就调用 web_search_tool。\n"
        "- 如果用户是在问某个项目、文件或上一轮内容里的“搜索功能怎么实现”，这是文件/代码理解任务，不是联网搜索任务。\n"
        "- 如果用户要求查找本地项目文件，就调用 find_tool。\n"
        "- 如果用户要求查看、阅读、总结某个具体本地文件，就调用 read_file_tool。\n"
        "- 工具返回结果后，必须以最新工具结果作为最高优先级证据；如果它与历史回答或你的先验冲突，明确纠正旧说法，再给最终答案。\n"
        f"- 当前模型按 {MAX_MODEL_LEN} tokens 上下文预算运行，最终回答输出上限为 {LLM_MAX_TOKENS} tokens；不要为了微小增益反复调用工具。\n"
        f"- 本轮最多执行 {MAX_TOTAL_TOOL_CALLS} 次工具调用、最多 {MAX_TOOL_TURNS} 个模型工具回合；达到上限时必须基于已有信息总结。\n"
        f"- 同一个工具最多只能连续调用 {MAX_CONSECUTIVE_TOOL_CALLS} 次；如果连续达到上限，也要基于已经掌握的信息给出阶段性回答，不要无限重试。\n"
        "- 只要可用工具能够满足用户请求，就不要声称自己无法使用工具。"
    )


def get_tool_bound_model():
    """返回绑定了当前工具 schema 的模型。"""
    return get_model().bind_tools(_get_tools())


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


def _with_no_think(text: str) -> str:
    if not LLM_NO_THINK:
        return text
    stripped = text.lstrip()
    if stripped.startswith("/no_think"):
        return text
    return f"/no_think\n{text}"


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


WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[^\s，。；;：:\n\r]+")


def _looks_like_context_followup(user_text: str) -> bool:
    """Deprecated: context routing now lives in backend.context_manager."""
    from backend.context_manager import looks_like_context_followup

    return looks_like_context_followup(user_text)


def _recent_context_paths(history: list[dict]) -> list[str]:
    """Deprecated: use backend.context_manager.recent_context_paths."""
    return recent_context_paths(history)


def _history_context_summary(history: list[dict]) -> str:
    """Deprecated: use backend.context_manager.history_context_summary."""
    return history_context_summary(history)


def _build_missing_tool_router_prompt(user_text: str, history: list[dict]) -> list[Any]:
    tools = _get_tools()
    tool_lines = []
    for item in tools:
        name = getattr(item, "name", getattr(item, "__name__", "tool"))
        description = (getattr(item, "description", None) or getattr(item, "__doc__", "") or "").strip()
        args = getattr(item, "args", None)
        args_text = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else "{}"
        tool_lines.append(f"- {name}: {description}\n  参数结构：{args_text}")

    context_summary = history_context_summary(history) or "无结构化历史上下文。"
    system = (
        "你是 BigDog 的工具路由器，只负责判断是否需要补发一次工具调用。\n"
        "背景：主模型刚才没有返回正式 tool_call。你要根据用户问题、历史上下文和可用工具，判断是否应该调用一个工具。\n"
        "只允许从可用工具列表里选择一个工具；如果不需要工具或信息不足，返回 tool_name 为 null。\n"
        "不要写解释，不要输出 Markdown，只输出 JSON。"
    )
    user = (
        f"可用工具列表：\n{chr(10).join(tool_lines)}\n\n"
        f"历史上下文：\n{context_summary}\n\n"
        f"用户问题：\n{user_text}\n\n"
        "返回格式：\n"
        '{"tool_name": "工具名或null", "args": {"参数名": "参数值"}}\n\n'
        "路由原则：\n"
        "1. 如果用户是在追问上一轮文件、上一轮工具结果或上一轮结论，优先选择能读取/检查该上下文的工具。\n"
        "2. 如果用户明确需要外部、在线、最新或公共网页信息，选择能搜索互联网的工具。\n"
        "3. 如果用户要查找本地路径或项目文件，选择能查找本地文件的工具。\n"
        "4. 如果用户只是普通聊天、写作或基于已有上下文已经能回答，返回 null。\n"
        "5. 参数必须尽量具体，不要用空对象敷衍。"
    )
    return [SystemMessage(content=system), HumanMessage(content=_with_no_think(user))]


def _parse_missing_tool_router_response(raw_text: str) -> dict | None:
    text = raw_text.strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)

    try:
        data = json.loads(text)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    tool_name = data.get("tool_name")
    if tool_name is None or str(tool_name).strip().lower() in {"", "null", "none", "no_tool"}:
        return None

    available = {getattr(item, "name", getattr(item, "__name__", "")) for item in _get_tools()}
    tool_name = str(tool_name).strip()
    if tool_name not in available:
        return None

    args = data.get("args")
    if not isinstance(args, dict):
        args = {}
    return {"name": tool_name, "args": args}


async def _route_missing_tool_call(user_text: str, history: list[dict]) -> dict | None:
    context_call = context_followup_tool_call(user_text, history)
    if context_call is not None:
        return context_call

    try:
        async with asyncio.timeout(20):
            _debug_log("主模型未返回 tool_call，正在请求通用工具路由器")
            result = await get_model().ainvoke(_build_missing_tool_router_prompt(user_text, history))
            _debug_dump_model_return("通用工具路由器返回", result)
    except Exception as e:
        _debug_log(f"通用工具路由器失败：{e}")
        return None

    raw = _message_reasoning_to_text(result) + _message_content_to_text(getattr(result, "content", ""))
    return _parse_missing_tool_router_response(raw)


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
    elif tool_name == "web_search_tool":
        trace["retrieved_chunks"].append(
            {
                "filename": "web_search_tool",
                "text": content[:1000],
            }
        )


def _build_partial_tool_answer(user_text: str, trace: dict, stop_reason: str | None = None) -> str:
    calls = trace.get("tool_calls", [])
    useful_chunks = []
    for item in trace.get("retrieved_chunks", []):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if text.startswith("工具执行失败") or text.startswith("网络搜索失败") or text.startswith("网络搜索没有返回内容"):
            continue
        useful_chunks.append((str(item.get("filename", "工具结果")), text))

    conclusion = _build_tool_fallback_conclusion(user_text, trace, useful_chunks, calls)
    reason = stop_reason or "工具循环已停止。"
    lines = [
        conclusion,
        "",
        f"说明：{reason}下面是我基于已拿到的工具结果整理出的依据。",
        "",
        f"用户问题：{user_text}",
        "",
    ]
    if useful_chunks:
        lines.append("主要依据：")
        for index, (source, text) in enumerate(useful_chunks[:5], start=1):
            preview = text[:500].strip()
            lines.append(f"{index}. 来源：{source}\n{preview}")
    else:
        lines.append("目前没有拿到可靠的工具结果，所以只能给出有限结论。")

    if calls:
        lines.append("")
        lines.append("已尝试的工具调用：")
        for index, call in enumerate(calls[-5:], start=1):
            tool_name = call.get("tool_name", "tool")
            args = call.get("args", {})
            preview = str(call.get("output_preview", "")).strip()
            if preview:
                preview = preview[:220]
            else:
                preview = "无返回内容"
            lines.append(f"{index}. {tool_name} 参数={args}，结果摘要：{preview}")

    lines.append("")
    lines.append(_build_tool_fallback_next_step(trace, useful_chunks))
    return "\n".join(lines)


def _build_tool_fallback_conclusion(
    user_text: str,
    trace: dict,
    useful_chunks: list[tuple[str, str]],
    calls: list[dict],
) -> str:
    """Build a user-facing conclusion when the model stops after tool use without final content."""
    matched_files = [str(item) for item in trace.get("matched_files", []) if str(item).strip()]
    read_files = [str(item) for item in trace.get("read_files", []) if str(item).strip()]
    tool_names = [str(call.get("tool_name", "")) for call in calls]
    lowered_question = user_text.lower()

    if matched_files or read_files:
        project_hint_files = matched_files + read_files
        blog_indicators = [
            path
            for path in project_hint_files
            if any(
                token in Path(path).name.lower()
                for token in ["blog", "post", "article", "db.json", "deploy_config", "favicon"]
            )
        ]
        config_indicators = [
            path
            for path in project_hint_files
            if Path(path).name.lower() in {"package.json", "vite.config.js", "vue.config.js", "next.config.js", "nuxt.config.js"}
        ]

        if "博客" in user_text or "blog" in lowered_question:
            if blog_indicators:
                examples = "、".join(Path(path).name for path in blog_indicators[:4])
                return f"结论：这是一个博客/站点项目。工具已经在目录里找到了 {examples} 这类典型博客数据、部署或站点资源文件；当前还没有读取核心配置或 README，所以后续可以继续解释它的技术栈和运行方式。"
            return "结论：目前的搜索结果不足以证明它是不是个人博客项目；需要继续读取 README、package.json 或站点配置文件才能确认。"

        if config_indicators:
            examples = "、".join(Path(path).name for path in config_indicators[:4])
            return f"结论：工具已经找到项目配置线索，例如 {examples}；可以据此继续读取文件来判断项目结构和运行方式。"

        examples = "、".join(Path(path).name for path in project_hint_files[:5])
        return f"结论：工具已经在目标目录中找到文件线索，例如 {examples}；目前能确认目录可访问，并且下一步应读取关键文件来解释项目。"

    if "web_search_tool" in tool_names:
        search_text = "\n".join(text for _, text in useful_chunks)
        titles = []
        for line in search_text.splitlines():
            stripped = line.strip()
            if re.match(r"^\d+\.\s+", stripped):
                titles.append(re.sub(r"^\d+\.\s+", "", stripped))
        if titles:
            examples = "；".join(titles[:3])
            return f"结论：网络搜索已经返回了一些可用结果，优先相关的结果包括：{examples}。下面给出依据摘要。"
        return "结论：已经尝试网络搜索，但结果不足以形成可靠答案；下面保留已拿到的信息和下一步建议。"

    if useful_chunks:
        return "结论：工具已经返回了部分有效信息，但模型没有完成最终整合；我先基于这些信息给出阶段性总结。"

    return "结论：这轮工具调用没有拿到足够有效的信息，因此暂时不能给出可靠判断。"


def _build_tool_fallback_next_step(trace: dict, useful_chunks: list[tuple[str, str]]) -> str:
    matched_files = [str(item) for item in trace.get("matched_files", []) if str(item).strip()]
    if matched_files:
        likely_files = [
            path
            for path in matched_files
            if Path(path).name.lower() in {"readme.md", "package.json", "bloginfolist.json", "db.json", "deploy_config.json"}
        ]
        if likely_files:
            return "下一步：建议继续读取这些关键文件：" + "、".join(likely_files[:5])
        return "下一步：建议读取 README、package.json、配置文件或数据文件，才能把“找到了什么”进一步解释成“这个项目怎么工作”。"

    if not useful_chunks:
        return "下一步：需要换一个更明确的路径、关键词，或检查工具/网络是否可用。"

    return "下一步：如果需要更完整的答案，可以继续读取最相关的文件或打开搜索结果来源核对。"


def _build_tool_loop_messages(user_text: str, history: list[dict]) -> list[Any]:
    tools = _get_tools()
    system_prompt = _build_tool_loop_system_prompt(tools)
    context_summary = _history_context_summary(history)
    if context_summary:
        system_prompt += "\n\n" + context_summary
    messages: list[Any] = [SystemMessage(content=system_prompt)]
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(("user", content))
        elif role == "assistant":
            messages.append(("ai", content))
        elif role == "assistant_tool_call":
            tool_name = str(msg.get("tool_name") or "tool")
            tool_args = msg.get("args") if isinstance(msg.get("args"), dict) else {}
            tool_call_id = str(msg.get("tool_call_id") or f"history_{len(messages)}")
            messages.append(AIMessage(
                content=str(content or ""),
                tool_calls=[{"name": tool_name, "args": tool_args, "id": tool_call_id}],
            ))
        elif role == "tool":
            tool_name = str(msg.get("tool_name") or "tool")
            tool_call_id = str(msg.get("tool_call_id") or f"history_{len(messages)}")
            messages.append(ToolMessage(content=str(content or ""), tool_call_id=tool_call_id, name=tool_name))
        elif role == "context":
            messages.append(("user", f"<历史工具上下文>\n{content}\n</历史工具上下文>"))
    messages.append(("user", _with_no_think(user_text)))
    return messages


def _tool_call_id(call: Any, fallback: str) -> str:
    if isinstance(call, dict):
        return str(call.get("id") or fallback)
    return str(getattr(call, "id", "") or fallback)


async def _execute_tool_call(call: Any) -> tuple[str, dict, str]:
    tool_name = _tool_call_name(call)
    args = _tool_call_args(call)
    tool_by_name = {getattr(item, "name", getattr(item, "__name__", "")): item for item in _get_tools()}
    selected = tool_by_name.get(tool_name)
    if selected is None:
        return tool_name, args, f"工具不存在：{tool_name}"

    try:
        result = await asyncio.to_thread(selected.invoke, args)
        return tool_name, args, _message_content_to_text(result) if not isinstance(result, str) else result
    except Exception as e:
        return tool_name, args, f"工具执行失败：{e}"


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
        "如果最新工具上下文与历史回答或先验冲突，必须以最新工具上下文为准并明确纠正旧说法。"
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
    return [SystemMessage(content=system), HumanMessage(content=_with_no_think(user))]


def _build_route_prompt(user_text: str) -> list[Any]:
    return [
        SystemMessage(
            content=(
                "你是请求路由器，只能输出一个中文词：工具 或 直接回答。\n"
                "可用工具名单：find_tool, read_file_tool, web_search_tool。\n"
                "输出 工具：用户要求搜索、查找、读取文件、查看项目、README、代码、路径、联网搜索、网络搜索、搜索一下、最新信息，或明确提到任一工具名。\n"
                "输出 直接回答：普通闲聊、身份问题、解释概念、写作润色，且不需要任何工具。\n"
                "示例：使用web_search_tool进行搜索 => 工具\n"
                "示例：网络搜索大狗大狗叫叫叫是什么东西 => 工具\n"
                "示例：查找 README => 工具\n"
                "示例：你是什么模型 => 直接回答\n"
                "不要解释，不要输出标点。"
            )
        ),
        HumanMessage(content=_with_no_think(f"用户请求：\n{user_text}\n\n路由结果：")),
    ]


def _looks_like_tool_request(user_text: str) -> bool:
    """用确定性规则兜住明确的工具意图，避免中文模型在路由阶段误判。"""
    text = user_text.lower()
    explicit_tool_names = ["find_tool", "read_file_tool", "web_search_tool"]
    if any(name in text for name in explicit_tool_names):
        return True

    tool_keywords = [
        "搜索",
        "搜一下",
        "查找",
        "查询",
        "联网",
        "网络搜索",
        "网页搜索",
        "最新",
        "读取文件",
        "阅读文件",
        "查看文件",
        "查看项目",
        "代码",
        "readme",
    ]
    return any(keyword in text for keyword in tool_keywords)


async def _route_query(user_text: str) -> str:
    if _looks_like_tool_request(user_text):
        return "tools"

    try:
        async with asyncio.timeout(ROUTE_TIMEOUT_SECONDS):
            _debug_log("即将请求路由模型")
            result = await get_model().ainvoke(_build_route_prompt(user_text))
            _debug_dump_model_return("路由模型返回", result)
    except Exception:
        return "direct_chat"

    raw = _message_reasoning_to_text(result) + _message_content_to_text(result.content)
    answer = _split_local_llm_output(raw)[1].lower()
    if "工具" in answer or re.search(r"\btools\b", answer):
        return "tools"
    return "direct_chat"


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
    messages.append(HumanMessage(content=_with_no_think(user_text)))
    return messages


async def _stream_model_answer(messages: list[Any]):
    raw_output = ""
    _debug_log(f"即将开始流式请求大模型，消息数量={len(messages)}")
    async for chunk in get_model().astream(messages):
        _debug_dump_model_return("流式模型返回片段", chunk)
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
    """Zleap 风格工具循环：模型每轮都拿到 tools schema，自行决定是否 tool_call。"""
    trace = {
        "tool_used": False,
        "tool_name": None,
        "tool_calls": [],
        "searched_paths": [],
        "matched_files": [],
        "read_files": [],
        "retrieved_chunks": [],
        "retrieval_stage": "tool_call_harness",
        "retrieval_mode": "native_tool_loop",
        "model_name": os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        "max_model_len": MAX_MODEL_LEN,
        "max_output_tokens": LLM_MAX_TOKENS,
        "max_tool_turns": MAX_TOOL_TURNS,
        "max_total_tool_calls": MAX_TOTAL_TOOL_CALLS,
        "max_consecutive_tool_calls": MAX_CONSECUTIVE_TOOL_CALLS,
    }
    messages = _build_tool_loop_messages(user_text, history)
    model = get_tool_bound_model()
    continue_nudges = 0
    last_tool_name = None
    consecutive_tool_calls = 0
    total_tool_calls = 0
    forced_tool_used = False

    yield {"type": "rag_step", "step": {"label": "正在装载工具上下文...", "icon": "🧰"}}

    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            for turn_index in range(MAX_TOOL_TURNS):
                yield {
                    "type": "rag_step",
                    "step": {
                        "label": f"模型回合 {turn_index + 1}",
                        "icon": "🧠",
                        "detail": "已向模型提供结构化工具列表。",
                    },
                }
                _debug_log(f"即将请求工具循环模型，第{turn_index + 1}轮，消息数量={len(messages)}")
                response = await model.ainvoke(messages)
                _debug_dump_model_return(f"工具循环模型返回-第{turn_index + 1}轮", response)
                if not isinstance(response, AIMessage):
                    raw_answer = _message_content_to_text(getattr(response, "content", ""))
                    yield {"type": "trace", "rag_trace": trace}
                    if raw_answer:
                        yield {"type": "content", "content": raw_answer}
                    return

                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    forced_call = None if forced_tool_used or trace.get("tool_used") else await _route_missing_tool_call(user_text, history)
                    if forced_call is not None:
                        forced_tool_used = True
                        forced_tool_name = _tool_call_name(forced_call)
                        forced_args = _tool_call_args(forced_call)
                        reason = f"模型没有发出工具调用，通用工具路由器选择调用 {forced_tool_name}。"
                        yield {
                            "type": "rag_step",
                            "step": {
                                "label": f"按显式意图调用工具：{forced_tool_name}",
                                "icon": "🔎",
                                "detail": str(forced_args),
                            },
                        }
                        messages.append(AIMessage(content="", tool_calls=[{"name": forced_tool_name, "args": forced_args, "id": f"forced_{turn_index}"}]))
                        executed_name, executed_args, content = await _execute_tool_call(forced_call)
                        total_tool_calls += 1
                        _append_tool_result_to_trace(trace, executed_name, executed_args, content)
                        messages.append(ToolMessage(content=content, tool_call_id=f"forced_{turn_index}", name=executed_name))
                        yield {
                            "type": "rag_step",
                            "step": {
                                "label": f"{executed_name} 返回结果",
                                "icon": "📄",
                                "detail": reason + "\n" + content[:180],
                            },
                        }
                        continue

                    raw_output = _message_reasoning_to_text(response) + _message_content_to_text(response.content)
                    thinking, answer = _split_local_llm_output(raw_output)
                    if answer:
                        yield {"type": "trace", "rag_trace": trace}
                        if thinking:
                            yield {"type": "thinking", "content": thinking}
                        yield {"type": "content", "content": answer}
                        return

                    if continue_nudges < 1:
                        continue_nudges += 1
                        messages.append(response)
                        messages.append(HumanMessage(content="你已经停止调用工具，但还没有给出最终回答。请以最新工具结果为最高优先级证据，若它与历史回答冲突先纠正旧说法，再用中文回答用户。"))
                        continue

                    reason = "模型没有继续调用工具，也没有生成最终回答；我先返回目前掌握的信息。"
                    yield {"type": "trace", "rag_trace": trace}
                    yield {"type": "content", "content": _build_partial_tool_answer(user_text, trace, reason)}
                    return

                messages.append(response)
                for index, call in enumerate(tool_calls):
                    call_id = _tool_call_id(call, f"call_{turn_index}_{index}")
                    tool_name = _tool_call_name(call)
                    args = _tool_call_args(call)
                    if total_tool_calls >= MAX_TOTAL_TOOL_CALLS:
                        reason = f"本轮工具调用已达到总上限 {MAX_TOTAL_TOOL_CALLS} 次，我先停止继续调用工具。"
                        yield {
                            "type": "rag_step",
                            "step": {
                                "label": "工具调用达到总上限",
                                "icon": "🛑",
                                "detail": reason,
                            },
                        }
                        yield {"type": "trace", "rag_trace": trace}
                        yield {"type": "content", "content": _build_partial_tool_answer(user_text, trace, reason)}
                        return

                    if tool_name == last_tool_name:
                        consecutive_tool_calls += 1
                    else:
                        last_tool_name = tool_name
                        consecutive_tool_calls = 1

                    if consecutive_tool_calls > MAX_CONSECUTIVE_TOOL_CALLS:
                        reason = f"{tool_name} 已连续调用 {MAX_CONSECUTIVE_TOOL_CALLS} 次，我先停止继续调用这个工具。"
                        yield {
                            "type": "rag_step",
                            "step": {
                                "label": f"{tool_name} 连续调用达到上限",
                                "icon": "🛑",
                                "detail": reason,
                            },
                        }
                        yield {"type": "trace", "rag_trace": trace}
                        yield {"type": "content", "content": _build_partial_tool_answer(user_text, trace, reason)}
                        return

                    yield {
                        "type": "rag_step",
                        "step": {
                            "label": f"调用工具：{tool_name}",
                            "icon": "🔎",
                            "detail": str(args),
                        },
                    }
                    executed_name, executed_args, content = await _execute_tool_call(call)
                    total_tool_calls += 1
                    _append_tool_result_to_trace(trace, executed_name, executed_args, content)
                    messages.append(ToolMessage(content=content, tool_call_id=call_id, name=executed_name))
                    yield {
                        "type": "rag_step",
                        "step": {
                            "label": f"{executed_name} 返回结果",
                            "icon": "📄",
                            "detail": content[:180],
                        },
                    }
                    if total_tool_calls >= MAX_TOTAL_TOOL_CALLS:
                        reason = f"本轮工具调用已达到总上限 {MAX_TOTAL_TOOL_CALLS} 次，我先停止继续调用工具。"
                        yield {
                            "type": "rag_step",
                            "step": {
                                "label": "工具调用达到总上限",
                                "icon": "🛑",
                                "detail": reason,
                            },
                        }
                        yield {"type": "trace", "rag_trace": trace}
                        yield {"type": "content", "content": _build_partial_tool_answer(user_text, trace, reason)}
                        return

                    if consecutive_tool_calls >= MAX_CONSECUTIVE_TOOL_CALLS:
                        reason = f"{executed_name} 已连续调用 {MAX_CONSECUTIVE_TOOL_CALLS} 次，我先停止继续调用这个工具。"
                        yield {
                            "type": "rag_step",
                            "step": {
                                "label": f"{executed_name} 连续调用达到上限",
                                "icon": "🛑",
                                "detail": reason,
                            },
                        }
                        yield {"type": "trace", "rag_trace": trace}
                        yield {"type": "content", "content": _build_partial_tool_answer(user_text, trace, reason)}
                        return

            yield {"type": "trace", "rag_trace": trace}
            yield {"type": "content", "content": _build_partial_tool_answer(user_text, trace)}
    except asyncio.TimeoutError:
        yield {"type": "error", "content": f"工具循环超时：超过 {AGENT_TIMEOUT_SECONDS}s，已停止本轮执行"}
        yield {"type": "trace", "rag_trace": trace}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield {"type": "error", "content": f"工具循环出错: {str(e)}"}
        yield {"type": "trace", "rag_trace": trace}
