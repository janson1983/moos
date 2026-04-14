import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, AsyncGenerator

# 将项目根目录添加到 sys.path 中，解决直接运行本脚本时找不到 'core' 模块的问题
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from fastapi import APIRouter, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from core.graph import build_graph

# 初始化全局的 LangGraph
# 这里只初始化一次，确保它能复用内部的 Checkpointer（记忆体）
graph = build_graph()

# 初始化 FastAPI 路由
router = APIRouter(prefix="/v1/agent", tags=["Agent"])

# 请求和响应模型
class RunAgentRequest(BaseModel):
    task_id: str
    message: str

class RunAgentResponse(BaseModel):
    status: str
    task_id: str

@router.post("/run")
async def run_agent(request: RunAgentRequest) -> RunAgentResponse:
    """
    (可选) 异步启动任务的接口，这里暂只做占位符。
    因为我们将核心逻辑放在了 `/stream` 接口里，以便建立长连接并实时推送内容。
    """
    return RunAgentResponse(status="Task queued (or use /stream directly)", task_id=request.task_id)

async def _event_generator(task_id: str, message: str) -> AsyncGenerator[str, None]:
    """
    负责运行 LangGraph 的迭代过程，并将中间结果生成 Server-Sent Events (SSE) 流
    """
    # 建立 LangGraph 配置，用 task_id 作为线程 id 持久化会话
    config = {"configurable": {"thread_id": task_id}}
    
    # 构建当前周期的输入状态
    state_input = {
        "messages": [HumanMessage(content=message)]
    }
    
    try:
        # LangGraph.astream 逐节点吐出执行状态
        async for output in graph.astream(state_input, config=config):
            for node_name, state_update in output.items():
                
                # 如果这个节点有内部思考步骤 (internal_steps) 需要推送到前端
                if "internal_steps" in state_update:
                    for step in state_update["internal_steps"]:
                        event_data = {
                            "type": "thought",
                            "node": node_name,
                            "content": step
                        }
                        # SSE 格式: `data: {...}\n\n`
                        yield f"data: {json.dumps(event_data)}\n\n"
                        # 可选：稍作停顿，使前端展示更平滑
                        await asyncio.sleep(0.05)
                        
                # 检查最新追加的消息 (如果是 executor/tool_node 生成的)
                if "messages" in state_update and state_update["messages"]:
                    last_msg = state_update["messages"][-1]
                    
                    # 1. 检查是不是 Executor 决定要调用工具 (Tool Calls)
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            event_data = {
                                "type": "tool_call",
                                "node": node_name,
                                "tool_name": tool_call.get("name"),
                                "tool_args": tool_call.get("args")
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                            
                    # 2. 检查是不是 Tool_Node 的工具执行结果 (Observation)
                    elif last_msg.type == "tool":
                        event_data = {
                            "type": "observation",
                            "node": node_name,
                            "tool_name": last_msg.name,
                            "result": last_msg.content
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                        
        # 整个大图执行结束，提取出最终的结果
        final_state = graph.get_state(config)
        messages = final_state.values.get("messages", [])
        
        final_answer = ""
        # 找到最后一条 AI 生成的非 Tool 的文本消息
        for msg in reversed(messages):
            if msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
                final_answer = msg.content
                break
                
        if not final_answer:
            final_answer = "Task executed but no direct textual response was generated."
            
        event_data = {
            "type": "result",
            "content": final_answer
        }
        yield f"data: {json.dumps(event_data)}\n\n"
        
    except Exception as e:
        error_data = {
            "type": "error",
            "content": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        
    # SSE 流结束标志
    yield "data: [DONE]\n\n"

@router.get("/stream/{task_id}")
async def stream_agent(task_id: str, message: str, request: Request):
    """
    提供 SSE 接口：
    例如 GET /v1/agent/stream/task_001?message=帮我分析一下文件
    """
    return StreamingResponse(
        _event_generator(task_id, message),
        media_type="text/event-stream"
    )

# 组装 FastAPI APP
app = FastAPI(title="MOOS API", version="1.0.0")
app.include_router(router)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    """处理前端上传的文件并保存到 workspace 目录下"""
    try:
        # 确保 workspace 目录存在
        workspace_dir = Path("workspace")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # 安全处理文件名：去除可能的路径穿越
        safe_filename = Path(file.filename).name
        file_path = workspace_dir / safe_filename
        
        # 保存文件内容
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        return JSONResponse(status_code=200, content={
            "status": "success", 
            "message": f"文件 {safe_filename} 上传成功",
            "file_path": f"workspace/{safe_filename}"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"文件上传失败: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    # 为了方便测试，可直接使用 python api/main.py 启动
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
