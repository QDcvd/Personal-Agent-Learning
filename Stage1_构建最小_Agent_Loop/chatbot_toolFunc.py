from openai import OpenAI  # 导入 OpenAI SDK（DeepSeek 兼容该接口）
import json                # 用于序列化/反序列化工具调用参数
import subprocess          # 用于执行 Linux grep / find / ls 等命令
import pathlib             # 用于跨平台文件搜索（Windows / Linux 均适用）
from dotenv import load_dotenv  # 从 .env 文件加载环境变量
import os                  # 读取环境变量

load_dotenv()  # 加载 .env 文件（含 DEEPSEEK_API_KEY）


def grep_search(pattern: str, path: str = ".") -> str:
    """
    工具函数：在 Linux 文件系统中递归搜索文件内容（封装 grep 命令）。
    Args:
        pattern: 搜索的正则表达式模式。
        path:    搜索的起始目录，默认为当前目录。
    Returns:
        匹配结果文本，最多返回前 2000 个字符。
    """
    try:
        # 执行: grep -rn --include=*.py <pattern> <path>
        # -r: 递归  -n: 显示行号  --include=*.py: 只搜 Python 文件
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, path],
            capture_output=True,  # 捕获标准输出
            text=True,            # 以文本形式返回
            timeout=60            # 超时 10 秒，防止卡死
        )
        if result.returncode == 0:
            # grep 返回 0 表示找到了匹配
            return result.stdout[:2000] or f"找到匹配，但内容为空: {pattern}"
        else:
            # grep 返回 1 表示无匹配，返回 2 表示错误
            return f"未找到匹配: {pattern}"
    except subprocess.TimeoutExpired:
        return f"搜索超时: {pattern}"
    except Exception as e:
        return f"搜索出错: {e}"


# grep 工具的 OpenAI Tool Schema 定义
# 这告诉模型：有一个叫 grep_search 的工具可以用，参数是什么
grep_tool = {
    "type": "function",  # 固定值，表示这是一个函数调用工具
    "function": {
        "name": "grep_search",        # 工具名称，必须与上面的 Python 函数名一致
        "description": "在 Linux 或 wsl 系统中递归搜索文件内容，支持正则表达式",  # 告诉模型何时使用
        "parameters": {               # 参数 schema（JSON Schema 格式）
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "搜索的正则表达式模式，如 'def main' 或 'import os'"
                },
                "path": {
                    "type": "string",
                    "description": "搜索的目录路径，默认为当前目录 '.'",
                    "default": "."
                }
            },
            "required": ["pattern"]   # pattern 是必填参数，path 可选
        }
    }
}

def find_search(pattern: str) -> str:
    try:
        result = subprocess.run(
            ["find", "/", "-name", pattern],
            capture_output=True,  # 捕获标准输出
            text=True,            # 以文本形式返回
            timeout=60            # 超时 10 秒，防止卡死
        )
        if result.returncode == 0:
            # grep 返回 0 表示找到了匹配
            return result.stdout[:2000] or f"找到匹配，但内容为空: {pattern}"
        else:
            # grep 返回 1 表示无匹配，返回 2 表示错误
            return f"未找到匹配: {pattern}"
    except subprocess.TimeoutExpired:
        return f"搜索超时: {pattern}"
    except Exception as e:
        return f"搜索出错: {e}"


find_tool = {
    "type": "function",  # 固定值，表示这是一个函数调用工具
    "function": {
        "name": "find_search",        # 工具名称，必须与上面的 Python 函数名一致
        "description": "在 Linux 或 wsl 系统中寻找特定的文件路径，注意忽略掉Permission denied的部分",  # 告诉模型何时使用
        "parameters": {               # 参数 schema（JSON Schema 格式）
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "一般你可以这样使用：find / -name xxxx.py"
                },
            },
            "required": ["pattern"]   # pattern 是必填参数，path 可选
        }
    }
}

def ll_search(pattern: str) -> str:
    try:
        result = subprocess.run(
            ["ls", pattern],
            capture_output=True,  # 捕获标准输出
            text=True,            # 以文本形式返回
            timeout=10            # 超时 10 秒，防止卡死
        )
        if result.returncode == 0:
            # grep 返回 0 表示找到了匹配
            return result.stdout[:2000] or f"找到匹配，但内容为空: {pattern}"
        else:
            # grep 返回 1 表示无匹配，返回 2 表示错误
            return f"未找到匹配: {pattern}"
    except subprocess.TimeoutExpired:
        return f"搜索超时: {pattern}"
    except Exception as e:
        return f"搜索出错: {e}"


ll_tool = {
    "type": "function",  # 固定值，表示这是一个函数调用工具
    "function": {
        "name": "ll_search",        # 工具名称，必须与上面的 Python 函数名一致
        "description": "在 Linux 或 wsl 系统中寻找特定的文件目录内的文件",  # 告诉模型何时使用
        "parameters": {               # 参数 schema（JSON Schema 格式）
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "一般你可以这样使用：ll；或者ll /root/xxxx"
                },
            },
            "required": ["pattern"]   # pattern 是必填参数，path 可选
        }
    }
}


def windows_search(pattern: str, root_dir: str = ".") -> str:
    """
    工具函数：在 Windows 文件系统中递归搜索文件（使用 pathlib 实现，跨平台兼容）。
    在 WSL 中可搜索 /e/ /c/ 等挂载的 Windows 盘符路径。
    Args:
        pattern: 文件名模式，支持通配符，如 "*.py"、"*test*"、"main*.*"
        root_dir: 搜索的根目录，默认为当前目录。
    Returns:
        匹配的文件路径列表，每行一个。
    """
    try:
        root = pathlib.Path(root_dir)  # 将字符串路径转为 pathlib 对象
        if not root.exists():          # 检查目录是否存在
            return f"目录不存在: {root_dir}"
        # rglob 递归匹配所有文件名符合 pattern 的文件
        # 按文件大小排序（从小到大），便于快速浏览
        matches = sorted(root.rglob(pattern), key=lambda p: p.stat().st_size)
        if not matches:
            return f"未找到匹配 {pattern} 的文件: {root_dir}"
        # 格式化结果：每行显示文件大小和路径
        lines = []
        for p in matches[:50]:  # 最多返回 50 条，防止输出过长
            size = p.stat().st_size  # 文件大小（字节）
            lines.append(f"{size:>8,d} B  {p}")
        result = "\n".join(lines)
        # 如果结果太多，提示被截断
        if len(matches) > 50:
            result += f"\n... 共 {len(matches)} 个匹配，仅显示前 50 个"
        return result
    except PermissionError as e:
        return f"权限不足，无法访问: {e}"
    except Exception as e:
        return f"搜索出错: {e}"


windows_tool = {
    "type": "function",
    "function": {
        "name": "windows_search",
        "description": "在 Windows 或 WSL 文件系统中递归搜索文件名（支持通配符），"
                       "可搜索 E:\\ AI_agent_Learning\\ 等目录下的文件",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "文件名模式，支持通配符。例如：*.py、*test*、main*.*"
                },
                "root_dir": {
                    "type": "string",
                    "description": "搜索的根目录路径，默认为当前目录。"
                                   "WSL 中 Windows 路径示例：/e/AI_agent_Learning",
                    "default": "."
                }
            },
            "required": ["pattern"]
        }
    }
}


class SimpleChatbot:
    """最小智能体：支持多轮对话 + 工具调用（含工具调用循环）"""

    def __init__(
        self,
        api_key: str = None,               # DeepSeek / OpenAI API 密钥，默认从 .env 读取
        model: str = None,                 # 模型名称，默认从 .env 读取
        system_prompt: str = None,         # 可选的系统提示词
        tools: list = None                 # 可选的工具定义列表
    ):
        """初始化聊天机器人，支持自定义模型、系统提示词和工具。"""
        # 创建 OpenAI 兼容客户端，注意 base_url 指向 DeepSeek 的 API 地址
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.messages = []       # 对话历史，存储所有消息
        # 如果有 system_prompt，作为第一条消息注入对话
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.tools = tools or []  # 工具定义列表，默认为空

        # 构建工具名 → 函数对象的映射表，用于后续执行工具
        # key = 工具名 (字符串)，value = 对应的 Python 函数
        self._tool_map = {}
        for t in self.tools:
            name = t["function"]["name"]  # 从 schema 中取出工具名
            # 从全局作用域中找到同名函数（要求工具函数定义在模块顶层）
            func = globals().get(name)
            if func:
                self._tool_map[name] = func

    @staticmethod
    def _clean(text: str) -> str:
        """
        移除字符串中的无效 surrogate 字符。
        某些终端输入或模型输出可能包含 UTF-8 不允许的代理字符，
        直接传给 API 会触发 UnicodeEncodeError。
        """
        if isinstance(text, str):
            # encode("utf-8", errors="replace") 将无效字符替换为 �
            # 再 decode 回字符串，就安全了
            return text.encode("utf-8", errors="replace").decode("utf-8")
        return text

    def chat(self, user_input: str, json_mode: bool = False,
             show_thinking: bool = False) -> str:
        """
        与模型对话的主方法。
        如果 self.tools 非空，会自动进入工具调用循环：
        模型返回 tool_calls → 执行工具 → 结果送回模型 → 直到模型直接回复文本。

        Args:
            user_input:   用户输入的文本。
            json_mode:    若为 True，强制模型以 JSON 格式输出。
            show_thinking: 若为 True，流式输出模型回复，实时打印到终端。
        Returns:
            模型的最终回复文本。
        """
        # 1. 清洗用户输入，防止非法字符
        user_input = self._clean(user_input)
        # 2. 将用户输入追加到对话历史
        self.messages.append({"role": "user", "content": user_input})

        # 3. 构造 API 请求参数
        kwargs = {
            "model": self.model,      # 模型名
            "messages": self.messages, # 完整的对话历史
        }
        # 4. 如果注册了工具，把工具定义传给 API
        if self.tools:
            kwargs["tools"] = self.tools
        # 5. 如果启用 JSON 模式，强制模型输出合法 JSON
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # 6. 工具调用循环：
        #    只要模型返回了 tool_calls，就执行工具并将结果送回模型，
        #    直到模型直接返回文本（没有 tool_calls）才跳出循环。
        while True:
            # 6a. 调用 API（分流式/非流式两种模式）
            if show_thinking:
                # ---------- 流式模式：逐 chunk 打印思考过程 ----------
                print("🤔 ", end="", flush=True)
                stream = self.client.chat.completions.create(
                    **kwargs, stream=True,
                    stream_options={"include_usage": True}
                )
                # 累积变量
                collected_content = ""               # 累积文本回复
                collected_tool_calls = {}             # 累积工具调用 {index: {id, name, arguments}}
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # ① 普通文本：实时打印
                    if delta.content:
                        collected_content += delta.content
                        print(delta.content, end="", flush=True)

                    # ② DeepSeek 推理链（reasoning_content 字段）
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        # 灰色斜体显示推理过程，与普通回复区分
                        print(f"\033[90;3m{delta.reasoning_content}\033[0m", end="", flush=True)

                    # ③ 工具调用：流式下是分片传来的，需累积拼接
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index  # 并行 tool_call 的索引
                            if idx not in collected_tool_calls:
                                collected_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.id:
                                collected_tool_calls[idx]["id"] += tc.id
                            if tc.function:
                                if tc.function.name:
                                    collected_tool_calls[idx]["name"] += tc.function.name
                                if tc.function.arguments:
                                    collected_tool_calls[idx]["arguments"] += tc.function.arguments

                print()  # 流式结束后换行

                # 如果有工具调用，打印决策信息
                if collected_tool_calls:
                    print(f"\033[33m🔧 模型决定调用 {len(collected_tool_calls)} 个工具:\033[0m")
                    for idx in sorted(collected_tool_calls.keys()):
                        tc = collected_tool_calls[idx]
                        print(f"\033[33m   {tc['name']}({tc['arguments']})\033[0m")

                # 将流式累积结果转成与非流式兼容的 msg 对象
                # 用 SimpleNamespace 模拟 API 返回的消息结构
                from types import SimpleNamespace as SN
                if collected_tool_calls:
                    tool_calls_list = []
                    for idx in sorted(collected_tool_calls.keys()):
                        tc = collected_tool_calls[idx]
                        tool_calls_list.append(SN(
                            id=tc["id"],
                            type="function",
                            function=SN(
                                name=tc["name"],
                                arguments=tc["arguments"]
                            )
                        ))
                    msg = SN(content=collected_content or None, tool_calls=tool_calls_list)
                else:
                    msg = SN(content=collected_content, tool_calls=None)
                # ---------- 流式模式结束 ----------

            else:
                # ---------- 非流式模式：原有逻辑 ----------
                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message

            # 6b. 检查模型是否调用了工具
            if not msg.tool_calls:
                # 没有 tool_calls → 这是最终文本回复
                # 清洗后存入历史并返回
                reply = self._clean(msg.content)
                self.messages.append({"role": "assistant", "content": reply})
                return reply

            # 6c. 有 tool_calls → 将 assistant 消息（含 tool_calls）存入历史
            if not show_thinking:
                # 非流式模式：打印工具决策
                print(f"\033[33m🔧 模型决定调用 {len(msg.tool_calls)} 个工具:\033[0m")
                for tc in msg.tool_calls:
                    print(f"\033[33m   {tc.function.name}({tc.function.arguments})\033[0m")

            # 转成纯 dict 再存历史，确保后续 API 调用时 JSON 可序列化
            if isinstance(msg, SN):
                # 流式模式：SimpleNamespace → dict
                tc_list = []
                for tc in msg.tool_calls:
                    tc_list.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                self.messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tc_list
                })
            else:
                # 非流式模式：API 原生对象（本身可 JSON 序列化）
                self.messages.append(msg)

            # 6d. 遍历所有工具调用，逐一执行
            for tc in msg.tool_calls:
                # 从 tool_call 中取出函数名和参数（JSON 字符串）
                func_name = tc.function.name
                # 将 JSON 字符串解析为 Python 字典
                func_args = json.loads(tc.function.arguments)
                # 从映射表中查找对应的函数
                func = self._tool_map.get(func_name)
                if func:
                    # 执行工具函数，将参数字典解包为关键字参数
                    print(f"\033[32m⚡ 执行: {func_name}({tc.function.arguments})\033[0m")
                    result = func(**func_args)
                else:
                    result = {"error": f"未知工具: {func_name}"}
                    print(f"\033[31m❌ 未知工具: {func_name}\033[0m")

                # 确保结果是字符串（如果不是就转 JSON）
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False)
                # 截断过长结果，避免刷屏
                display_result = result[:500] + "..." if len(result) > 500 else result
                print(f"\033[32m📦 结果: {display_result}\033[0m")

                # 6e. 将工具执行结果以 role="tool" 的形式送回给模型
                self.messages.append({
                    "role": "tool",          # 固定值：工具结果
                    "tool_call_id": tc.id,    # 关联到对应的 tool_call
                    "content": result         # 工具执行结果文本
                })

            # 6f. 更新 kwargs 中的 messages 为最新的历史（含工具结果）
            #     然后继续循环，让模型基于工具结果生成下一步
            kwargs["messages"] = self.messages


# ===== 主程序入口 =====
if __name__ == '__main__':
    # 创建 chatbot 实例（api_key 自动从 .env 读取 DEEPSEEK_API_KEY）
    chat = SimpleChatbot(
        system_prompt="你是一个文件搜索助手。你可以使用以下工具帮助用户搜索文件：\n"
        system_prompt="你是一个文件搜索助手。你可以使用以下工具帮助用户搜索文件：\n"
                       "1. grep_search — 在 Linux/WSL 中搜索文件内容（正则表达式）\n"
                       "2. find_search — 在 Linux/WSL 中按文件名搜索文件\n"
                       "3. ll_search — 查看 Linux/WSL 目录内容\n"
                       "4. windows_search — 在 Windows/WSL 中按文件名通配符搜索文件，支持 /e/ /c/ 等路径",
        tools=[grep_tool, find_tool, ll_tool, windows_tool]  # 注册所有工具
    )

    # 交互式对话循环
    while True:
        # 获取用户输入
        user_input = input('请输入对话：')
        # 调用 chat 方法（开启 show_thinking 查看思考过程）
        reply = chat.chat(user_input, show_thinking=True)
        # 打印回复
        print("回复：" + reply + "\n")
    #     # 打印回复
    #     print("回复：" + reply + "\n")
