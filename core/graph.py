import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from schema.state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from core.config import get_llm, WORKSPACE_DIR, MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)

import subprocess

# ... (WORKSPACE_DIR 初始化和 get_llm 函数保持原样)

@tool
def read_file(file_path: str) -> str:
    """读取指定路径的文件内容。当用户要求分析或查看某个文件时，使用此工具读取它。只能读取 workspace 目录下的文件。"""
    try:
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = WORKSPACE_DIR / target_path
        target_path = target_path.resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only read files within {WORKSPACE_DIR}"

        if not target_path.exists():
            return f"Error: File {file_path} does not exist."
            
        if not target_path.is_file():
            return f"Error: {file_path} is a directory, not a file."

        if target_path.stat().st_size > 1 * 1024 * 1024:
            return f"Error: File {file_path} is too large (exceeds 1MB). Please request smaller files or split them."

        with open(target_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

@tool
def edit_file(file_path: str, content: str) -> str:
    """写入或覆盖指定路径的文件内容。当用户要求新增或修改某个文件时使用。只能修改 workspace 目录下的文件。"""
    try:
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = WORKSPACE_DIR / target_path
        target_path = target_path.resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only edit/create files within {WORKSPACE_DIR}"

        # 确保父目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Success: File {file_path} has been written."
    except Exception as e:
        return f"Error writing file {file_path}: {str(e)}"

@tool
def delete_file(file_path: str) -> str:
    """删除指定路径的文件。当用户明确要求删除某个文件时使用。只能删除 workspace 目录下的文件。"""
    try:
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = WORKSPACE_DIR / target_path
        target_path = target_path.resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only delete files within {WORKSPACE_DIR}"

        if not target_path.exists():
            return f"Error: File {file_path} does not exist."

        if target_path.is_dir():
            return f"Error: {file_path} is a directory. This tool can only delete files."

        target_path.unlink()
        return f"Success: File {file_path} has been deleted."
    except Exception as e:
        return f"Error deleting file {file_path}: {str(e)}"

@tool
def execute_shell(command: str) -> str:
    """执行 Shell 脚本或命令。由于安全原因，执行目录将被强制限制在 workspace 下。任何试图通过命令（如 `cd ..` 或绝对路径）修改外部文件的行为都可能受限。"""
    try:
        # 在 workspace 目录下执行命令
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            timeout=30  # 限制执行时间防止死循环脚本
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]:\n{result.stderr}"
        
        if not output.strip():
            output = "Command executed successfully with no output."
            
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool
def search_files(regex_pattern: str, file_pattern: str = "*", directory: str = ".") -> str:
    """
    使用正则表达式在指定目录及匹配特定文件名模式的文件中搜索内容。
    参数：
    - regex_pattern: 要搜索的正则表达式。
    - file_pattern: 文件名匹配模式，例如 '*.py', '*.ts'。默认为 '*' (所有文件)。
    - directory: 相对 workspace 的子目录路径。默认为 '.' (根目录)。
    返回匹配的内容及行号。
    """
    import fnmatch
    import re
    
    try:
        base_dir = WORKSPACE_DIR / directory
        base_dir = base_dir.resolve()
        
        if not str(base_dir).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only search within {WORKSPACE_DIR}"
            
        if not base_dir.exists() or not base_dir.is_dir():
            return f"Error: Directory {directory} does not exist."

        compiled_regex = re.compile(regex_pattern)
        results = []
        
        # 限制最多搜索 500 个文件或返回 1000 行结果，防止结果过大
        files_scanned = 0
        matches_found = 0
        
        for root, _, files in os.walk(base_dir):
            for filename in files:
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue
                    
                files_scanned += 1
                if files_scanned > 500:
                    results.append("... (Search aborted: Too many files scanned. Please narrow your search directory or file_pattern.)")
                    break
                    
                filepath = Path(root) / filename
                
                # 跳过太大的文件或二进制文件
                try:
                    if filepath.stat().st_size > 1024 * 1024:
                        continue
                        
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        file_matches = []
                        for i, line in enumerate(lines):
                            if compiled_regex.search(line):
                                file_matches.append(f"{i+1}: {line.strip()}")
                                matches_found += 1
                                
                                if matches_found > 1000:
                                    break
                                    
                        if file_matches:
                            rel_path = filepath.relative_to(WORKSPACE_DIR)
                            results.append(f"--- {rel_path} ---")
                            results.extend(file_matches)
                            results.append("")
                            
                except UnicodeDecodeError:
                    pass # 跳过非文本文件
                    
                if matches_found > 1000:
                    results.append("... (Search aborted: Too many matches found. Please refine your regex_pattern.)")
                    break
            else:
                continue
            break
            
        if not results:
            return f"No matches found for pattern '{regex_pattern}' in '{file_pattern}' files under '{directory}'."
            
        return "\n".join(results)
    except Exception as e:
        return f"Error searching files: {str(e)}"

@tool
def replace_in_file(file_path: str, diff: str) -> str:
    """
    修改已存在文件中的特定内容段落。
    参数：
    - file_path: 相对于 workspace 的文件路径。
    - diff: SEARCH/REPLACE 差异块，格式必须完全遵循以下规范：
      ------- SEARCH
      [你要替换掉的精确原文（包括完整的行、空格、缩进）]
      =======
      [你想替换成的新内容]
      +++++++ REPLACE
    注意：SEARCH 块必须精确匹配文件中的原始内容，不能有任何多余或遗漏。此工具只会替换找到的第一个匹配项。
    """
    try:
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = WORKSPACE_DIR / target_path
        target_path = target_path.resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only edit files within {WORKSPACE_DIR}"

        if not target_path.exists() or not target_path.is_file():
            return f"Error: File {file_path} does not exist."

        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 简单的 SEARCH/REPLACE 解析逻辑
        search_marker = "------- SEARCH"
        mid_marker = "======="
        replace_marker = "+++++++ REPLACE"
        
        blocks = diff.split(search_marker)
        
        if len(blocks) < 2:
            return "Error: Invalid diff format. Missing '------- SEARCH' marker."
            
        success_count = 0
        new_content = content
        
        for block in blocks[1:]: # 跳过第一个空串
            if mid_marker not in block or replace_marker not in block:
                continue
                
            parts = block.split(mid_marker)
            search_text = parts[0].strip("\n")
            
            replace_parts = parts[1].split(replace_marker)
            replace_text = replace_parts[0].strip("\n")
            
            if not search_text:
                continue
                
            # 严格检查 SEARCH 块是否存在于文件中
            if search_text not in new_content:
                return f"Error: The SEARCH block could not be found exactly as provided in {file_path}. Please make sure whitespace and indentations match perfectly."
                
            # 只替换第一个匹配项
            new_content = new_content.replace(search_text, replace_text, 1)
            success_count += 1
            
        if success_count == 0:
            return "Error: No valid SEARCH/REPLACE blocks were found and executed."

        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        return f"Success: Replaced {success_count} section(s) in {file_path}."
    except Exception as e:
        return f"Error replacing in file {file_path}: {str(e)}"

# 注册可用的工具
tools = [read_file, edit_file, delete_file, execute_shell, search_files, replace_in_file]
tools_by_name = {t.name: t for t in tools}

def _truncate_history(messages: list) -> list:
    """保留最近的若干条对话，防止 Token 上下文溢出"""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    # 始终保留第一条请求（通常是任务定义），然后保留最新的 MAX_HISTORY_MESSAGES - 1 条
    return [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]

import json

async def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor / Planner 节点：负责将用户目标拆解为任务列表，或分配给多个并发 Agent。
    """
    messages = state.get("messages", [])
    current_plan = state.get("current_plan", "")
    
    if not messages:
        return {"next_step": END, "internal_steps": ["[Planner] No messages found."]}
        
    llm = get_llm()
    system_prompt = SystemMessage(content='''You are a brilliant Planner Agent. Your job is to analyze the user's request.
If the request requires or explicitly asks for MULTIPLE different roles/agents to process the same task concurrently (e.g., "use different roles", "parallel agents"), you MUST output ONLY a valid JSON array of tasks. 
Each JSON object must have 'role' and 'task' keys. Example:
[
  {"role": "Financial Analyst", "task": "Analyze the financial impact..."},
  {"role": "Risk Manager", "task": "Evaluate potential risks..."}
]

Otherwise, if it is a standard sequential task, output a concise step-by-step numbered plan in plain text.
''')
    
    history_messages = _truncate_history(messages)
    conversation_context = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages if msg.content])
    
    try:
        response = await llm.ainvoke([
            system_prompt, 
            HumanMessage(content=f"Here is the conversation history:\n{conversation_context}\n\nPlease generate the plan or JSON array of roles based on the latest context.")
        ])
        current_plan = response.content
        
        # 尝试解析 JSON 判断是否为并发多 Agent 任务
        text_content = current_plan.strip()
        if text_content.startswith("```json"):
            text_content = text_content[7:-3].strip()
        
        worker_tasks = []
        next_step = "executor"
        try:
            parsed = json.loads(text_content)
            if isinstance(parsed, list) and len(parsed) > 0 and "role" in parsed[0] and "task" in parsed[0]:
                worker_tasks = parsed
                next_step = "parallel_workers"
                step_msg = f"Planner assigned tasks to {len(worker_tasks)} parallel agents."
            else:
                step_msg = "Planner dynamically updated the plan based on full context."
        except json.JSONDecodeError:
            step_msg = "Planner dynamically updated the plan based on full context."

    except Exception as e:
        logger.error(f"Planner LLM call failed: {e}")
        step_msg = f"Planner failed to generate plan due to error: {e}"
        worker_tasks = []
        next_step = "executor"
        if not current_plan:
            current_plan = "Error: Plan generation failed."
        
    return {
        "current_plan": current_plan,
        "worker_tasks": worker_tasks,
        "next_step": next_step,
        "internal_steps": [f"[Planner] {step_msg}"]
    }

async def parallel_workers_node(state: AgentState) -> Dict[str, Any]:
    """
    并发执行多个 Agent 任务的节点。
    """
    worker_tasks = state.get("worker_tasks", [])
    if not worker_tasks:
        return {"next_step": "executor"}
    
    llm = get_llm()
    
    async def run_worker(worker):
        role = worker.get("role", "Expert")
        task = worker.get("task", "")
        prompt = SystemMessage(content=f"You are an expert with the role: {role}. Your task is: {task}. Please provide your detailed analysis and solution. Be direct and professional.")
        try:
            res = await llm.ainvoke([prompt])
            return f"--- [{role}] Report ---\n{res.content}"
        except Exception as e:
            return f"--- [{role}] Report ---\nError: {e}"
            
    tasks = [run_worker(w) for w in worker_tasks]
    results = await asyncio.gather(*tasks)
    
    return {
        "worker_results": results,
        "next_step": "summarizer",
        "internal_steps": [f"[Parallel Workers] {len(worker_tasks)} expert agents have completed their tasks concurrently."]
    }

async def summarizer_node(state: AgentState) -> Dict[str, Any]:
    """
    汇总多个 Agent 结果的节点。
    """
    results = state.get("worker_results", [])
    messages = state.get("messages", [])
    
    llm = get_llm()
    sys_prompt = SystemMessage(content="You are a Master Summarizer Agent. Several expert agents have analyzed the user's request from different perspectives. Your job is to synthesize their findings into a single, cohesive, and comprehensive final conclusion. Format it beautifully with Markdown.")
    
    user_req = messages[-1].content if messages else "Unknown request."
    combined_results = "\n\n".join(results)
    
    prompt = HumanMessage(content=f"Original Request: {user_req}\n\nExpert Reports:\n{combined_results}\n\nPlease provide the final summary and conclusion.")
    
    try:
        res = await llm.ainvoke([sys_prompt, prompt])
        final_message = res
    except Exception as e:
        final_message = AIMessage(content=f"Error summarizing results: {e}")
        
    return {
        "messages": [final_message],
        "next_step": END,
        "internal_steps": ["[Summarizer] Master agent synthesized all reports into the final conclusion."],
        "worker_tasks": [], 
        "worker_results": []
    }

async def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Executor 节点：根据计划调用 Tool 或直接生成回复。
    """
    # 检查是否刚经历了用户审批 (Approval 回调过来的)
    if state.get("awaiting_approval"):
        return {
            "next_step": "tool_node",
            "awaiting_approval": False, # 已经确认过了，放行
            "internal_steps": ["[Executor] User approved the sensitive operations. Proceeding to execution."]
        }

    messages = state.get("messages", [])
    current_plan = state.get("current_plan", "")
    
    history_messages = _truncate_history(messages)
    
    llm = get_llm().bind_tools(tools)
    # 优化点 1：防止 Tool 执行错误导致的无限死循环
    system_prompt = SystemMessage(content=f"""You are an Executor Agent. Here is your current plan:
{current_plan}

Execute the plan based on the conversation history. 
If the user provides a file path or asks to analyze a file, ALWAYS use the `read_file` tool. 
If you need to find a function, class, or pattern across multiple files, use the `search_files` tool first.
If asked to modify an EXISTING file with targeted changes, strongly prefer the `replace_in_file` tool. ONLY use `edit_file` when creating a brand new file or completely overwriting a small file.
If asked to delete a file, use the `delete_file` tool.
If asked to run a script or command, use the `execute_shell` tool.
Remember that the workspace directory is strictly isolated. All file operations and shell commands MUST be confined within it.
If you need more information, ask the user directly.
Once you have the information, provide the final answer to the user.

CRITICAL INSTRUCTION: If a tool returned an error in the previous steps, analyze the cause and correct your parameters. DO NOT repeat the exact same tool call if it failed.
""")
    
    try:
        response = await llm.ainvoke([system_prompt] + list(history_messages))
        
        # 判断是否需要调用工具
        if getattr(response, "tool_calls", None):
            # HITL 检查：是否包含危险工具 (删除文件、执行命令)
            sensitive_tools = ["delete_file", "execute_shell"]
            needs_approval = any(tc['name'] in sensitive_tools for tc in response.tool_calls)
            
            if needs_approval:
                # 触发拦截：暂停在这个节点
                next_step = "await_approval"
                msg = f"Security Intercept: LLM requested sensitive tools ({[tc['name'] for tc in response.tool_calls if tc['name'] in sensitive_tools]}). Awaiting user approval."
                output_msg = response # 先把 tool_calls 存到 message 里
            else:
                next_step = "tool_node"
                msg = f"LLM decided to call tools: {[t['name'] for t in response.tool_calls]}"
                output_msg = response
        else:
            next_step = "reviewer"
            msg = "LLM generated a direct response to the user."
            output_msg = response
            
    except Exception as e:
        logger.error(f"Executor LLM call failed: {e}")
        next_step = "reviewer"
        msg = f"Executor failed due to API error: {e}"
        # 生成一个兜底回复，防止图崩溃
        output_msg = AIMessage(content=f"Sorry, I encountered an error while processing your request: {e}")
        
    return {
        "messages": [output_msg], # 追加响应记录
        "next_step": next_step,
        "internal_steps": [f"[Executor] {msg}"]
    }

async def execute_single_tool(tool_call, tools_by_name):
    """并发执行单个工具的辅助函数"""
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_instance = tools_by_name.get(tool_name)
    
    if tool_instance:
        try:
            # 优化点 4：使用 ainvoke 支持异步并发执行
            result = await tool_instance.ainvoke(tool_args)
            step_msg = f"Executed tool: {tool_name} successfully."
        except Exception as e:
            result = f"Error during tool execution: {e}"
            step_msg = f"Tool {tool_name} execution failed: {e}"
    else:
        result = f"Tool {tool_name} not found."
        step_msg = f"Failed to find tool: {tool_name}."
        
    return ToolMessage(
        content=str(result),
        name=tool_name,
        tool_call_id=tool_call.get("id", "")
    ), step_msg

async def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Tool 节点：并发执行具体工具调用。
    """
    messages = state.get("messages", [])
    if not messages:
        return {"next_step": "reviewer"}
        
    last_message = messages[-1]
    
    # 安全检查：确保是包含了工具调用的消息
    if not hasattr(last_message, "tool_calls"):
        return {"next_step": "executor"}
        
    # 优化点 4：并发执行所有请求的 tool call (使用 asyncio.gather)
    tasks = [execute_single_tool(tc, tools_by_name) for tc in last_message.tool_calls]
    results = await asyncio.gather(*tasks)
    
    tool_messages = [r[0] for r in results]
    step_msgs = [r[1] for r in results]
        
    return {
        "messages": tool_messages, 
        "next_step": "executor",   # 工具执行完后回到 executor 继续判断
        "internal_steps": [f"[Tool_Node] {msg}" for msg in step_msgs]
    }

async def reviewer_node(state: AgentState) -> Dict[str, Any]:
    """
    Reviewer/Reflector 节点：不仅检查是否完成，还负责决定是否需要重新规划
    """
    messages = state.get("messages", [])
    
    # 提取最后几条对话进行判断
    recent_messages = _truncate_history(messages)[-4:]
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages if msg.content])
    
    llm = get_llm()
    # 优化点 5：Reviewer 加入 LLM 判断，实现智能反射 (Reflector) 循环
    checker_prompt = f"根据以下最近的对话，判断用户的初始目标是否已完全达成，或者系统是否已经给出了最终明确的答复。如果已达成或已答复，回复 YES；如果认为还需要调用工具或重新规划才能完成，回复 NO。\n\n对话内容:\n{conversation_text}"
    
    try:
        res = await llm.ainvoke([HumanMessage(content=checker_prompt)])
        if "YES" in res.content.upper():
            return {"next_step": END, "internal_steps": ["[Reviewer] 目标已达成或已答复，结束当前执行流。"]}
        else:
            # 如果没完成，回到 planner 重新审视计划
            return {"next_step": "planner", "internal_steps": ["[Reviewer] 任务未完全结束，请求重新规划。"]}
    except Exception as e:
        logger.error(f"Reviewer LLM call failed: {e}")
        return {"next_step": END, "internal_steps": ["[Reviewer] Error checking completion status, pausing."]}

def should_continue(state: AgentState) -> str:
    """
    路由逻辑，根据 next_step 决定下一个节点
    """
    # 如果处于等待审批状态，直接中断流，图引擎在这里暂停
    if state.get("next_step") == "await_approval":
        return END
        
    return state.get("next_step", END)

def build_graph() -> StateGraph:
    """
    构建并返回 LangGraph 状态机，并附带记忆体用于持久化对话状态。
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("reviewer", reviewer_node)
    
    # 新增并发多 Agent 节点
    workflow.add_node("parallel_workers", parallel_workers_node)
    workflow.add_node("summarizer", summarizer_node)
    
    workflow.set_entry_point("planner")
    
    workflow.add_conditional_edges("planner", should_continue, {
        "executor": "executor", 
        "parallel_workers": "parallel_workers",
        END: END
    })
    
    # 并发任务流转
    workflow.add_conditional_edges("parallel_workers", should_continue, {"summarizer": "summarizer"})
    workflow.add_conditional_edges("summarizer", should_continue, {END: END})
    
    # 标准工具任务流转
    workflow.add_conditional_edges("executor", should_continue, {"tool_node": "tool_node", "reviewer": "reviewer", END: END})
    workflow.add_conditional_edges("tool_node", should_continue, {"executor": "executor", END: END})
    workflow.add_conditional_edges("reviewer", should_continue, {"planner": "planner", END: END})
    
    # 使用 MemorySaver 来支持断点续传和多轮对话状态持久化
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
