# Project Specification: MOOS (Micro-Orchestra OS)

## 1. 项目愿景
构建一个轻量级、可独立部署、基于 Web 调用且具有高度自主权（Autonomous）的多 Agent 管理系统。类似于服务端版本的 OpenClaw/Cline，能够通过自主思考、调用工具、自我修正来完成复杂任务。

---

## 2. 核心架构设计 (System Architecture)

### 2.1 技术栈 (Tech Stack)
* **Core Engine**: Python 3.10+, **LangGraph** (用于处理循环状态机与有向无环图)。
* **API Framework**: **FastAPI** (支持异步高并发与 WebSocket/SSE)。
* **Communication**: **SSE (Server-Sent Events)** (实时向 Web 端推送 Agent 的 Thought/Action 流)。
* **Sandbox**: 异步 `subprocess` 封装 (执行 Python/Shell)。
* **Storage**: SQLite (任务追踪) + 内存状态快照。

### 2.2 逻辑模型 (The ReAct Loop)
系统遵循 **Plan -> Action -> Observe -> Reflect** 循环：
1. **Planner**: 将用户目标拆解为任务列表。
2. **Executor**: 根据任务调用相应 Agent 和 Tools。
3. **Observer**: 捕获工具输出（stdout/stderr/API Result）。
4. **Reflector**: 判断是否达到预期，若失败则修正计划重新执行。

---

## 3. 模块化实施路线图 (Implementation Roadmap)

### Phase 1: 核心编排器定义 (The State Machine)
**Cline 任务目标：**
- 初始化项目结构：`core/`, `agents/`, `tools/`, `api/`, `schema/`。
- 定义 `AgentState` 类 (TypedDict)，包含 `messages`, `next_step`, `internal_steps`, `current_plan`。
- 使用 **LangGraph** 构建基础状态图：`START -> Planner -> Executor -> Tool_Node -> Reviewer -> END`。

### Phase 2: 自主工具链 (The Sandbox Tools)
**Cline 任务目标：**
- 实现 `BaseTool` 抽象基类。
- **TerminalTool**: 异步执行 Shell 命令，支持超时控制。
- **FileTool**: 限制在工作目录内的 `read`, `write`, `list_dir` 功能。
- **SearchTool**: 集成 Tavily 或 DuckDuckGo 用于联网搜索。

### Phase 3: 响应式 Web 接口 (FastAPI & SSE)
**Cline 任务目标：**
- 建立 `POST /v1/agent/run` 接口，启动异步 Task。
- 实现 `GET /v1/agent/stream/{task_id}`，通过 **SSE (Server-Sent Events)** 实时推送：
    - `event: thought` (Agent 的逻辑推理)
    - `event: tool_call` (工具名称与参数)
    - `event: observation` (工具执行结果)
    - `event: result` (任务最终产出)

### Phase 4: 记忆与断点续传 (Persistence)
**Cline 任务目标：**
- 实现 LangGraph 的 `Checkpointer`，将 Agent 状态持久化到 SQLite。
- 支持用户在 Web 端对特定步骤进行“确认”或“重试”（Human-in-the-loop）。

---

## 4. 完整的实现提示词 (Full Input Prompt)

如果您想使用 AI 编程助手（如 Cline / Cursor / GitHub Copilot）从零复刻出当前的完整项目，可以直接复制以下提示词作为最初的指令：

"你现在是一名顶尖的 AI 全栈架构师。请帮我实现一个名为 MOOS (Micro-Orchestra OS) 的极简多智能体调度系统。

**核心技术栈：**
- 后端：Python 3.10+, FastAPI, LangGraph, langchain-openai
- 前端：原生 HTML + CSS + JS (不需要 Vue/React)
- 通信机制：SSE (Server-Sent Events)

**架构与功能要求：**
1. **核心 Agent (core/graph.py)**:
   - 使用 `LangGraph` 构建一个状态图，包含四个节点：`Planner` (任务规划) -> `Executor` (工具调用决策) -> `Tool_Node` (并发执行工具) -> `Reviewer` (反思检查是否完成)。
   - 必须使用异步 (`async`) 并在执行过程中抛出中间的思考步骤 (`internal_steps`)。
2. **工具集 (Tools)**:
   - 实现四个基础工具：`read_file`, `edit_file`, `delete_file`, `execute_shell`。
   - **绝对安全限制**：所有文件的读取、修改、删除，以及 Shell 的执行，必须被严格限制在一个隔离的 `workspace/` 目录下，禁止任何 `../` 路径穿越攻击。
3. **配置文件 (core/config.py & .env)**:
   - 将 LLM 的 API 配置（Model, Base URL, API Key等）和系统配置（如 Workspace 路径、最大历史记录数）抽离到单独的 `core/config.py` 中，并通过 `python-dotenv` 读取项目根目录的 `.env` 文件。
4. **后端接口 (api/main.py)**:
   - 提供一个处理文件上传的 `/v1/upload` POST 接口（文件需存入 `workspace/`）。
   - 提供一个基于 SSE 的 `/v1/agent/stream/{task_id}` GET 接口，能够实时推送大模型的 Thought、Tool Call、Observation 和最终 Result。
   - 挂载 `static/` 目录提供前端静态文件服务。
5. **前端交互 (static/index.html)**:
   - 编写一个类似于 ChatGPT / Gemini 的暗色主题交互界面。
   - 实现**多文件暂存区 (Staging Area)**：支持一次多选或多次上传本地文件，文件以小卡片（带删除按钮）的形式显示在输入框上方。发送时自动组合路径和用户的 Prompt。
   - 能够解析后端的 SSE 事件流，并在界面上实时打字输出思考过程和结果。

请先提供核心实现的代码，要求代码极致精简、结构清晰、注释详尽。"

---

## 5. 关键安全与性能原则
- **Path Isolation**: 所有文件操作必须限制在指定的 `./workspace` 文件夹内。
- **Token Optimization**: 实现自动对历史消息进行 Summary，防止 context window 溢出。
- **Graceful Shutdown**: 确保中断任务时能正确杀掉子进程。