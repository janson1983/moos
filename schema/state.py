import operator
from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    核心 AgentState 类，用于 LangGraph 的状态管理。
    """
    # 消息列表，使用 operator.add 确保每次 append 新消息
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # 下一个要执行的节点或动作的名称
    next_step: str
    
    # 内部执行步骤，用于记录 Agent 的中间思考/行为，便于流式输出
    internal_steps: Annotated[list[str], operator.add]
    
    # 当前执行的任务计划
    current_plan: str
    
    # 任务标识符
    task_id: str
    
    # 多智能体并发任务列表 (每个元素包含 'role' 和 'task')
    worker_tasks: list[dict]
    
    # 多智能体并发执行的结果集合
    worker_results: list[str]
