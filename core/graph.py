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

# 注册可用的工具
tools = [read_file, edit_file, delete_file, execute_shell]
tools_by_name = {t.name: t for t in tools}

def _truncate_history(messages: list) -> list:
    """保留最近的若干条对话，防止 Token 上下文溢出"""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    # 始终保留第一条请求（通常是任务定义），然后保留最新的 MAX_HISTORY_MESSAGES - 1 条
    return [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]

async def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor / Planner 节点：负责将用户目标拆解为任务列表。
    """
    messages = state.get("messages", [])
    current_plan = state.get("current_plan", "")
    
    if not messages:
        return {"next_step": END, "internal_steps": ["[Planner] No messages found."]}
        
    llm = get_llm()
    system_prompt = SystemMessage(content="You are a brilliant Planner Agent. Your job is to break down the user's task into a step-by-step numbered plan. Keep it concise. Please read the conversation history and focus on the most recent user instructions to update or create the plan.")
    
    # 优化点 3：Planner 记忆隔离优化。将截断后的历史消息序列化给 Planner，使其能阅读整个上下文而不是只看第一句话。
    history_messages = _truncate_history(messages)
    conversation_context = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages if msg.content])
    
    try:
        response = await llm.ainvoke([
            system_prompt, 
            HumanMessage(content=f"Here is the conversation history:\n{conversation_context}\n\nPlease generate or update the plan based on the latest context.")
        ])
        current_plan = response.content
        step_msg = "Planner dynamically updated the plan based on full context."
    except Exception as e:
        logger.error(f"Planner LLM call failed: {e}")
        step_msg = f"Planner failed to generate plan due to error: {e}"
        if not current_plan:
            current_plan = "Error: Plan generation failed."
        
    return {
        "current_plan": current_plan,
        "next_step": "executor",
        "internal_steps": [f"[Planner] {step_msg}"]
    }

async def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Executor 节点：根据计划调用 Tool 或直接生成回复。
    """
    messages = state.get("messages", [])
    current_plan = state.get("current_plan", "")
    
    # 优化点 2：Token 溢出预防。截断历史，防止 Token 爆炸
    history_messages = _truncate_history(messages)
    
    llm = get_llm().bind_tools(tools)
    # 优化点 1：防止 Tool 执行错误导致的无限死循环
    system_prompt = SystemMessage(content=f"""You are an Executor Agent. Here is your current plan:
{current_plan}

Execute the plan based on the conversation history. 
If the user provides a file path or asks to analyze a file, ALWAYS use the `read_file` tool. 
If asked to create or modify a file, use the `edit_file` tool.
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
    
    workflow.set_entry_point("planner")
    
    workflow.add_conditional_edges("planner", should_continue, {"executor": "executor", END: END})
    workflow.add_conditional_edges("executor", should_continue, {"tool_node": "tool_node", "reviewer": "reviewer", END: END})
    workflow.add_conditional_edges("tool_node", should_continue, {"executor": "executor", END: END})
    workflow.add_conditional_edges("reviewer", should_continue, {"planner": "planner", END: END})
    
    # 使用 MemorySaver 来支持断点续传和多轮对话状态持久化
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
