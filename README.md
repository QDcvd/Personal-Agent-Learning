
# 🧠 AI Agent 学习路径

> 从零开始系统性学习 AI Agent 的实战项目集合。

---

## 📁 项目结构

```
AI_agent_Learning/
│
├── Stage0_Agent是什么/                    # 概念基础
│   ├── README.md                         # Chatbot / Workflow / Agent / Multi-Agent 概念详解
│   ├── Anthropic-Building-Effective-Agents-中文翻译.md  # Anthropic 工作流模式
│   ├── OpenAI-A-Practical-Guide-to-Building-Agents-中文翻译.md  # OpenAI Agent 构建指南
│   └── chatbot.py                        # 最小 Chatbot 实现
│
├── Stage1_构建最小_Agent_Loop/            # 动手实战
│   ├── README.md                         # 项目文档
│   └── chatbot_toolFunc.py               # 带工具调用的 Agent（4 个搜索工具 + 思考过程可视化）
│
└── README.md                             # ← 本文件
```

---

## 🗺️ 学习路线

```
Stage 0: 理解概念
├── 什么是 Chatbot / Workflow / Agent / Multi-Agent
├── Anthropic 的 5 种工作流模式
└── OpenAI 的 Agent 构建 3 要素（模型 + 工具 + 指令）
        │
        ▼
Stage 1: 动手实现
├── 最小工具调用循环（Tool Call Loop）
├── 4 个文件搜索工具
├── 流式思考可视化
└── JSON 结构化输出
        │
        ▼
Stage 2+: (待续...)
```

---

## 📚 各 Stage 说明

### Stage 0 — 概念基础

> 在写代码之前，先理解你正在构建什么。

| 文档 | 来源 | 核心内容 |
|------|------|---------|
| [Chatbot/Workflow/Agent/Multi-Agent 详解](Stage0_Agent是什么/README.md) | 自编 | 四个概念的定义、对比、选型决策树 |
| [Building Effective Agents](Stage0_Agent是什么/Anthropic-Building-Effective-Agents-中文翻译.md) | Anthropic | 5 种工作流模式、何时用 Agent、ACI 原则 |
| [A Practical Guide to Building Agents](Stage0_Agent是什么/OpenAI-A-Practical-Guide-to-Building-Agents-中文翻译.md) | OpenAI | 模型选择、工具定义、编排模式、护栏设计 |

### Stage 1 — 最小 Agent 循环

> 动手实现第一个带工具调用的 Agent。

- **446 行全注释代码**，逐行解释每一段的作用
- **4 个搜索工具**：`grep_search` / `find_search` / `ll_search` / `windows_search`
- **工具调用循环**：LLM 自主决策 → 执行工具 → 结果送回 → 继续推理
- **流式思考**：`show_thinking=True` 实时查看模型的思考过程
- **JSON 模式**：`json_mode=True` 强制模型输出结构化数据

---

## 🚀 开始学习

```bash
# 1. 从概念开始
open Stage0_Agent是什么/README.md

# 2. 动手运行代码
cd Stage1_构建最小_Agent_Loop
pip install openai
python chatbot_toolFunc.py
```

---

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| Python 3 | 核心语言 |
| OpenAI SDK | 兼容 DeepSeek / OpenAI 等 LLM API |
| subprocess | 调用 Linux 搜索命令（grep/find/ls） |
| pathlib | 跨平台文件搜索（Windows/Linux） |
| stream | 流式输出实现思考过程可视化 |
