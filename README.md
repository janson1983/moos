# MOOS (Micro-Orchestra OS)

MOOS 是一个基于 **LangGraph** 和 **FastAPI** 构建的**极简多智能体调度系统（Agent System）**。

> ⚠️ **声明**：这是一个**学习和实验性质的项目**。旨在通过最少量、最核心的代码，向开发者展示如何从零开始搭建一个具备**规划 (Planner) -> 执行 (Executor) -> 工具调用 (Tools) -> 反思 (Reviewer)** 闭环机制的智能 Agent 系统。

通过本项目，您可以快速了解：
- 如何利用 LangGraph 构建复杂的 Agent 工作流（StateGraph）。
- 如何实现 Server-Sent Events (SSE) 流式传输，将大模型的每一步思考和工具执行实时推送到前端。
- 如何在 Web 端构建一个类似 ChatGPT / Gemini 的多文件暂存、上传及长连接对话的交互界面。
- 如何为大模型安全地挂载本地操作工具（读取、修改、删除文件，甚至执行受限的 Shell 脚本）。

---

## 🌟 核心特性

- 🧠 **思维闭环**：内置 Planner、Executor、Tool Node、Reviewer 四大核心节点，支持动态目标拆解与自我纠错。
- 🚀 **并发多 Agent 架构 (Map-Reduce)**：当任务需要多视角或多角色分析时，Planner 可自动拆解子任务并分配给多个 Worker 并发执行，最后由 Master Summarizer 节点汇总结论，极大提升复杂任务的处理效率。
- ⚡ **SSE 流式输出**：前端实时感知 Agent 的内部执行步骤（Internal Steps）和调用状态，告别长时间的黑盒等待。
- 🛡️ **沙盒安全机制**：工具的执行范围被严格限制在 `workspace/` 目录下，防止大模型通过 `../` 或绝对路径破坏宿主机的系统文件。
- 📂 **多文件暂存区（Staging Area）**：原生支持多个文件的选中上传与可视化管理，结合 Prompt 一起发送给 Agent 进行分析。
- 🛠️ **核心能力**：自带文件读取、局部文件替换 (`replace_in_file`)、全量写入、文件删除、正则全局搜索 (`search_files`) 以及安全 Shell 执行等基础 Tool。模拟了主流 AI 编程助手（如 Cline, Cursor）的代码库探索与 AST 级别修改的核心能力。

---

## 📁 目录结构

系统代码保持极致精简，去除了冗余的抽象封装，方便阅读：

```text
moos/
├── api/
│   └── main.py          # FastAPI 服务入口，包含 /v1/upload 和 /v1/agent/stream SSE流式接口
├── core/
│   ├── config.py        # 系统全局配置与大模型 (LLM) 初始化设置
│   └── graph.py         # LangGraph 核心逻辑（节点定义、图的组装、工具函数的实现）
├── schema/
│   └── state.py         # LangGraph 的共享状态结构 (AgentState) 定义
├── static/
│   └── index.html       # 纯前端原生 HTML/JS 页面，实现了交互UI、文件上传和 SSE 监听
├── workspace/           # 沙盒工作区，Agent 操作文件的唯一合法目录（自动生成）
├── action.md            # 生成此项目的终极 Prompt (Prompt for AI Copilot) 💡
├── .env.example         # 环境变量配置模板
├── .gitignore           # Git 忽略配置
└── README.md            # 项目说明文件
```

> **💡 关于 `action.md`**：  
> 这个文件非常特别！它包含了**用于生成本系统全部基础代码的系统级 Prompt（提示词）**。如果您想学习如何指导 AI 编程助手（如 Cline, Cursor 或 Claude）写出一个完整的工程，可以直接打开 `action.md` 阅读。您可以把它复制下来喂给您的 AI 助手，它就能原封不动地为您复刻出这套框架！

---

## 🚀 快速开始

### 1. 克隆项目与安装依赖

```bash
git clone https://github.com/your-username/moos.git
cd moos

# 推荐使用虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装核心依赖
pip install fastapi uvicorn python-multipart langchain-openai langgraph python-dotenv
```

### 2. 配置环境变量

复制环境模板文件：
```bash
cp .env.example .env
```

打开 `.env` 文件，填入您的大模型配置（兼容 OpenAI 接口格式的模型均可，如 Qwen, DeepSeek, GPT-4 等）：
```ini
OPENAI_API_MODEL=你的模型名称 (如: qwen3.5-plus)
OPENAI_API_BASE=你的模型API_BASE_URL (如: https://api.openai.com/v1/)
OPENAI_API_KEY=你的模型API_KEY
```

### 3. 启动服务

```bash
python3 api/main.py
```
*或者使用 uvicorn 启动：*
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 浏览器访问

打开浏览器访问：[http://127.0.0.1:8000](http://127.0.0.1:8000)

现在，您可以：
1. 点击输入框左侧的 📎 图标，上传电脑里的几个本地文件。
2. 在输入框中下达指令，例如：“**请分析一下我刚上传的这几个文件，总结出它们的共通点，并将总结结果写到 workspace/summary.md 文件中。**”
3. 观察右侧 Agent 是如何一步步规划、读取、分析并最终执行写入操作的！

---

## 💡 进阶使用与二次开发

如果您希望在这个学习项目上进行扩展：

1. **添加新工具 (Tools)**：
   只需在 `core/graph.py` 中编写带 `@tool` 装饰器的 Python 函数，并将其加入底部的 `tools = [...]` 列表中即可。系统会自动将其绑定到 Executor 节点。
   
2. **并发多 Agent (Map-Reduce) 扩展**：
   当前系统已内置了 `parallel_workers` 和 `summarizer` 节点。您可以尝试让它扮演不同领域的专家对同一问题进行深度剖析，比如“请用三个不同的角色（财务分析师、技术总监、风险经理）同时帮我评估这个商业计划书，最后给出综合建议”。

3. **优化反思逻辑 (Reviewer)**：
   当前的 `reviewer_node` 利用简单的 LLM 提问来决定是否 "YES"（达成任务）或 "NO"（重新规划）。您可以引入更复杂的打分机制或专家审核链来强化任务完成度。

4. **修改前端样式**：
   前端采用原生 HTML+CSS+JS 编写，没有复杂的打包工具束缚。直接修改 `static/index.html` 刷新页面即可生效，非常适合快速魔改。

---

## 📄 许可证

MIT License. 欢迎任何人基于此项目进行学习、修改与发布！