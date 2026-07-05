"""Conversation context assembly for BigDog.

This module keeps context handling out of the FastAPI route and closer to the
Zleap style: session messages are projected into model-ready history entries,
and tool calls/results from previous turns stay attached to the conversation
instead of being reduced to plain assistant text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


MAX_HISTORY_MESSAGES = 12
MAX_TRACE_TOOL_CALLS = 6
MAX_TOOL_RESULT_CHARS = 1200
MAX_CONTEXT_PATHS = 8
WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[^\s，。；;：:\n\r]+")


def build_agent_history(session_messages: list[dict]) -> list[dict]:
    """Build model history from stored session messages.

    The old BigDog route only passed `{role, content}`. That lost the important
    part of a tool turn: which tool was called, with which arguments, and what it
    returned. This projector keeps recent human/assistant messages and expands
    recent `rag_trace` objects into synthetic tool-call/tool-result pairs.
    """
    recent_messages = session_messages[-MAX_HISTORY_MESSAGES:]
    history: list[dict] = []

    for index, msg in enumerate(recent_messages):
        msg_type = msg.get("type")
        content = str(msg.get("content") or "")
        if msg_type == "human":
            history.append({"role": "user", "content": content})
            continue

        if msg_type != "ai":
            continue

        history.append({
            "role": "assistant",
            "content": content,
            "rag_trace": msg.get("rag_trace"),
        })
        history.extend(_trace_to_history_entries(msg.get("rag_trace"), index))

    return history


def history_context_summary(history: list[dict]) -> str:
    paths = recent_context_paths(history)[:MAX_CONTEXT_PATHS]
    if not paths:
        return ""

    lines = [
        "上一轮可复用的结构化上下文：",
        "相关文件：",
        *[f"- {path}" for path in paths],
        "如果用户用“它/他/这个/那/上面/刚才”追问，优先理解为在追问这些文件、上一轮工具结果或上一轮结论。",
        "不要把文件名或上一轮主题里的“搜索/实现/功能”等词误判为新的联网搜索请求。",
    ]
    return "\n".join(lines)


def recent_context_paths(history: list[dict]) -> list[str]:
    paths: list[str] = []

    def add_path(value: Any) -> None:
        if not value:
            return
        path = str(value).strip().strip("`'\"")
        path = path.rstrip("。；;,，")
        if path and path not in paths:
            paths.append(path)

    for item in reversed(history[-MAX_HISTORY_MESSAGES:]):
        trace = item.get("rag_trace") or {}
        if isinstance(trace, dict):
            for key in ("read_files", "matched_files"):
                values = trace.get(key) or []
                if isinstance(values, list):
                    for value in reversed(values):
                        add_path(value)
            for chunk in reversed(trace.get("retrieved_chunks") or []):
                if isinstance(chunk, dict):
                    add_path(chunk.get("filename"))

        for key in ("path", "filename", "source"):
            add_path(item.get(key))

        content = str(item.get("content", ""))
        for match in reversed(WINDOWS_PATH_PATTERN.findall(content)):
            add_path(match)

    existing_files = [path for path in paths if Path(path).is_file()]
    return existing_files or paths


def context_followup_tool_call(user_text: str, history: list[dict] | None = None) -> dict | None:
    """Resolve context follow-ups to the most recent concrete context object.

    This module deliberately does not route generic tool intent such as web
    search. Tool selection belongs to the agent adapter, where the runtime knows
    the registered tool list and can use a generic router prompt.
    """
    text = user_text.strip()
    history = history or []

    if looks_like_context_followup(text):
        context_paths = recent_context_paths(history)
        if context_paths:
            return {"name": "read_file_tool", "args": {"path": context_paths[0]}}

    return None


def looks_like_context_followup(user_text: str) -> bool:
    text = user_text.strip()
    followup_markers = ["那", "那么", "他", "它", "这个", "这个文件", "刚才", "上面", "上述", "前面", "你刚"]
    question_markers = ["怎么", "如何", "为什么", "讲讲", "解释", "实现", "内容", "里面", "方式", "原理"]
    return any(marker in text for marker in followup_markers) and any(marker in text for marker in question_markers)


def _trace_to_history_entries(trace: Any, message_index: int) -> list[dict]:
    if not isinstance(trace, dict):
        return []

    entries: list[dict] = []
    tool_calls = trace.get("tool_calls") or []
    if isinstance(tool_calls, list):
        for index, call in enumerate(tool_calls[-MAX_TRACE_TOOL_CALLS:]):
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name") or call.get("name") or "tool")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            result = str(call.get("output_preview") or "").strip()
            call_id = f"history_{message_index}_{index}_{_safe_id(tool_name)}"
            entries.append({
                "role": "assistant_tool_call",
                "content": f"历史工具调用：{tool_name}",
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "args": args,
            })
            entries.append({
                "role": "tool",
                "content": _truncate(result or "历史工具调用没有记录返回内容。", MAX_TOOL_RESULT_CHARS),
                "tool_call_id": call_id,
                "tool_name": tool_name,
            })

    context_note = _trace_context_note(trace)
    if context_note:
        entries.append({
            "role": "context",
            "content": context_note,
            "rag_trace": trace,
        })
    return entries


def _trace_context_note(trace: dict) -> str:
    lines = ["历史工具上下文摘要："]
    matched_files = [str(path) for path in trace.get("matched_files") or [] if str(path).strip()]
    read_files = [str(path) for path in trace.get("read_files") or [] if str(path).strip()]
    searched_paths = trace.get("searched_paths") or []
    retrieved_chunks = trace.get("retrieved_chunks") or []

    if matched_files:
        lines.append("匹配文件：")
        lines.extend(f"- {path}" for path in matched_files[:MAX_CONTEXT_PATHS])
    if read_files:
        lines.append("已读取文件：")
        lines.extend(f"- {path}" for path in read_files[:MAX_CONTEXT_PATHS])
    if searched_paths:
        lines.append("搜索路径：")
        lines.append(_truncate(json.dumps(searched_paths[:4], ensure_ascii=False), 600))
    if isinstance(retrieved_chunks, list) and retrieved_chunks:
        lines.append("检索片段：")
        for chunk in retrieved_chunks[:4]:
            if not isinstance(chunk, dict):
                continue
            filename = str(chunk.get("filename") or "工具结果")
            text = str(chunk.get("text") or "").strip()
            if text:
                lines.append(f"- 来源：{filename}\n  摘要：{_truncate(text, 260)}")

    return "\n".join(lines) if len(lines) > 1 else ""


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", value).strip("_") or "tool"


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...(truncated)"
