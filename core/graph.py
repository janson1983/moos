import os
import asyncio
from pathlib import Path
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from schema.state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from core.config import get_llm, MAX_HISTORY_MESSAGES, setup_logger
from tools.builtin import tools, tools_by_name

logger = setup_logger("moos.graph")

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
        logger.info("[Planner] Requesting LLM for plan/role-allocation...")
        response = await llm.ainvoke([
            system_prompt, 
            HumanMessage(content=f"Here is the conversation history:\n{conversation_context}\n\nPlease generate the plan or JSON array of roles based on the latest context.")
        ])
        current_plan = response.content
        logger.debug(f"[Planner] Raw LLM response: {current_plan[:200]}...")
        
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
                logger.info(f"[Planner] Split into {len(worker_tasks)} concurrent Map-Reduce tasks.")
            else:
                step_msg = "Planner dynamically updated the plan based on full context."
        except json.JSONDecodeError:
            step_msg = "Planner dynamically updated the plan based on full context."

    except Exception as e:
        logger.exception(f"[Planner] LLM call or processing failed: {e}")
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
    
    logger.info(f"[Parallel Workers] Dispatching {len(worker_tasks)} workers concurrently.")
    async def run_worker(worker):
        role = worker.get("role", "Expert")
        task = worker.get("task", "")
        prompt = SystemMessage(content=f"You are an expert with the role: {role}. Your task is: {task}. Please provide your detailed analysis and solution. Be direct and professional.")
        try:
            logger.debug(f"[Worker:{role}] Started task.")
            res = await llm.ainvoke([prompt])
            logger.debug(f"[Worker:{role}] Finished task.")
            return f"--- [{role}] Report ---\n{res.content}"
        except Exception as e:
            logger.error(f"[Worker:{role}] Failed: {e}")
            return f"--- [{role}] Report ---\nError: {e}"
            
    tasks = [run_worker(w) for w in worker_tasks]
    results = await asyncio.gather(*tasks)
    logger.info(f"[Parallel Workers] All {len(worker_tasks)} workers completed.")
    
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
    
    logger.info("[Summarizer] Synthesizing parallel worker results...")
    try:
        res = await llm.ainvoke([sys_prompt, prompt])
        final_message = res
        logger.info("[Summarizer] Synthesis completed successfully.")
    except Exception as e:
        logger.exception(f"[Summarizer] Error summarizing results: {e}")
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
    logger.info("[Executor] Node started.")
    # 检查是否刚经历了用户审批 (Approval 回调过来的)
    if state.get("is_approved"):
        logger.info("[Executor] Recovering from user approval. Proceeding to tool_node.")
        return {
            "next_step": "tool_node",
            "is_approved": False, # 消耗掉批准标记
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
                sensitive_requested = [tc['name'] for tc in response.tool_calls if tc['name'] in sensitive_tools]
                logger.warning(f"[Executor] LLM requested sensitive tools: {sensitive_requested}. Triggering HITL intercept.")
                next_step = "await_approval"
                msg = f"Security Intercept: LLM requested sensitive tools ({sensitive_requested}). Awaiting user approval."
                output_msg = response # 先把 tool_calls 存到 message 里
                
                # 更新状态
                return {
                    "messages": [output_msg],
                    "next_step": next_step,
                    "awaiting_approval": True, # 标记为等待审批
                    "is_approved": False,
                    "internal_steps": [f"[Executor] {msg}"]
                }
            else:
                next_step = "tool_node"
                msg = f"LLM decided to call tools: {[t['name'] for t in response.tool_calls]}"
                logger.info(f"[Executor] {msg}")
                output_msg = response
        else:
            next_step = "reviewer"
            msg = "LLM generated a direct response to the user."
            logger.info("[Executor] Direct textual response generated, proceeding to Reviewer.")
            output_msg = response
            
    except Exception as e:
        logger.exception(f"[Executor] LLM call failed: {e}")
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
        logger.info(f"[Tool:{tool_name}] Executing with args: {tool_args}")
        try:
            # 优化点 4：使用 ainvoke 支持异步并发执行
            result = await tool_instance.ainvoke(tool_args)
            step_msg = f"Executed tool: {tool_name} successfully."
            logger.debug(f"[Tool:{tool_name}] Result: {str(result)[:200]}...")
        except Exception as e:
            logger.exception(f"[Tool:{tool_name}] Execution failed: {e}")
            result = f"Error during tool execution: {e}"
            step_msg = f"Tool {tool_name} execution failed: {e}"
    else:
        logger.error(f"[ToolNode] Requested unknown tool: {tool_name}")
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
    
    logger.info("[Reviewer] Checking task completion status...")
    try:
        res = await llm.ainvoke([HumanMessage(content=checker_prompt)])
        if "YES" in res.content.upper():
            logger.info("[Reviewer] Conclusion: Task COMPLETED. Ending flow.")
            return {"next_step": END, "internal_steps": ["[Reviewer] 目标已达成或已答复，结束当前执行流。"]}
        else:
            # 如果没完成，回到 planner 重新审视计划
            logger.info("[Reviewer] Conclusion: Task INCOMPLETE. Returning to planner.")
            return {"next_step": "planner", "internal_steps": ["[Reviewer] 任务未完全结束，请求重新规划。"]}
    except Exception as e:
        logger.exception(f"[Reviewer] LLM check failed: {e}")
        return {"next_step": END, "internal_steps": ["[Reviewer] Error checking completion status, pausing."]}

def should_continue(state: AgentState) -> str:
    """
    路由逻辑，根据 next_step 决定下一个节点
    """
    # 如果处于等待审批状态，直接中断流，图引擎在这里暂停
    if state.get("awaiting_approval"):
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
