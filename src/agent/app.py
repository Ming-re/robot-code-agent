"""FastAPI 服务：SSE 推 Agent 的执行轨迹，外加代码的 Y/N 确认。

    POST /api/chat          发起或恢复会话
    POST /api/code/confirm  保存代码，或者带意见继续
    GET  /api/health        健康检查
    GET  /                  前端页面

会话状态在 LangGraph 的检查点里，这里只存会话元信息。
客户端断连后要把 runner task cancel 掉，不然图会一直跑，token 白烧。

启动：python -m src.agent.app
"""

import asyncio
import contextlib
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# 项目根目录加入 sys.path, 保证以脚本方式（PyCharm Run）运行时也能导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agent.generator import CodeGenerator
from src.agent.graph import AGENT_SYSTEM_PROMPT, build_graph
from src.agent.tools import ToolExecutor
from src.config import (
    API_HOST,
    API_PORT,
    CHUNKS_JSON,
    COLLECTION_NAME,
    EMBEDDING_MODEL_DIR,
    GENERATED_CODE_DIR,
    MILVUS_URI,
    SESSION_TTL_SECONDS,
)
from src.embedding.bge_m3 import BgeM3Encoder
from src.milvus_store import MilvusStore

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_HEARTBEAT_SECONDS = 15


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)


class ConfirmRequest(BaseModel):
    session_id: str
    accept: bool
    feedback: str | None = None


class Session:
    """会话元信息。

    代码正文不在这里，以 LangGraph 检查点为唯一来源。
    这里只放确认接口要用的引用和去重状态。
    """

    def __init__(self, session_id):
        self.id = session_id
        self.requirement = ""
        self.created = time.monotonic()
        self.seen_calls: set[str] = set()


@asynccontextmanager
async def lifespan(application: FastAPI):
    generator = CodeGenerator()
    encoder = BgeM3Encoder(EMBEDDING_MODEL_DIR)
    store = MilvusStore(MILVUS_URI, COLLECTION_NAME)
    all_chunks = {c["chunk_id"]: c
                  for c in json.loads(Path(CHUNKS_JSON).read_text(
                      encoding="utf-8"))}
    executor = ToolExecutor(encoder, store, generator, all_chunks)
    application.state.graph = build_graph(executor)
    application.state.generator = generator
    application.state.store = store
    application.state.executor = executor
    yield


app = FastAPI(title="机器人指令 Agent（LangGraph）", lifespan=lifespan)
_sessions: dict[str, Session] = {}
_lock = asyncio.Lock()


def _purge_expired():
    """清理过期会话（原先 _sessions 永不清理，是内存泄漏点）。"""
    now = time.monotonic()
    expired = [s.id for s in _sessions.values()
               if now - s.created > SESSION_TTL_SECONDS]
    for sid in expired:
        _sessions.pop(sid, None)


def get_session(session_id) -> Session:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return session


def _sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """发起/恢复 Agent 会话: SSE 输出 Agent 执行轨迹。"""
    graph = app.state.graph
    async with _lock:
        _purge_expired()
        if req.session_id:
            session = get_session(req.session_id)
            is_new = False
        else:
            session = Session(uuid.uuid4().hex[:12])
            session.requirement = req.message.strip()
            _sessions[session.id] = session
            is_new = True

    thread_cfg = {"configurable": {"thread_id": session.id}}
    # 有挂起的 interrupt(ask_user 等待回答) 则恢复执行
    snapshot = await graph.aget_state(thread_cfg)
    has_interrupt = bool(snapshot.tasks and snapshot.tasks[0].interrupts)

    if is_new:
        graph_input = {
            "requirement": session.requirement,
            "messages": [SystemMessage(content=AGENT_SYSTEM_PROMPT),
                         HumanMessage(content=session.requirement)],
        }
    elif has_interrupt:
        graph_input = Command(resume=req.message.strip())
    else:
        # 无挂起: 视为用户追加意见, 继续同一线程
        graph_input = {"messages": [HumanMessage(content=req.message.strip())]}

    async def gen():
        # 首帧: 会话 ID（前端据此恢复后续交互）
        yield _sse({"type": "session", "session_id": session.id})
        queue: asyncio.Queue = asyncio.Queue()

        def emit(evt):
            # 同一 call_id 的事件只推一次（interrupt 重放会让工具事件重复）
            call_id = evt.get("call_id")
            if call_id:
                if call_id in session.seen_calls:
                    return
                session.seen_calls.add(call_id)
            queue.put_nowait(evt)

        cfg = {"configurable": {"thread_id": session.id, "emit": emit}}

        async def runner():
            try:
                async for update in graph.astream(
                        graph_input, config=cfg, stream_mode="updates"):
                    for node, upd in update.items():
                        if node != "agent" or "messages" not in upd:
                            continue
                        # 思考文本在此推送; 工具事件由 tools_node 执行时推送
                        for m in upd["messages"]:
                            if getattr(m, "content", ""):
                                emit({"type": "thought", "content": m.content})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                emit({"type": "error", "message": f"Agent 执行异常: {e}"})
            finally:
                queue.put_nowait(None)  # 结束哨兵

        async def heartbeat():
            """SSE 注释行保活: 代理/LB 通常 60s 空闲就断连。"""
            try:
                while True:
                    await asyncio.sleep(_HEARTBEAT_SECONDS)
                    queue.put_nowait({"type": "ping"})
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(runner())
        hb = asyncio.create_task(heartbeat())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if event.get("type") == "ping":
                    yield ": ping\n\n"   # SSE 注释行，客户端忽略
                    continue
                yield _sse(event)
        finally:
            # 客户端断连 / 正常结束: 都要收尸, 否则图会继续跑并继续烧 token
            hb.cancel()
            if not task.done():
                task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await hb

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # nginx 不缓冲，否则流式变成"一次性吐完"
        },
    )


@app.post("/api/code/confirm")
async def confirm(req: ConfirmRequest):
    """Y/N 确认: Y 保存代码; N 带意见则由前端再次调 /api/chat 继续会话。"""
    graph = app.state.graph
    async with _lock:
        session = get_session(req.session_id)

    # 单一真相源: 代码从检查点读, 不读 Session 镜像
    snapshot = await graph.aget_state(
        {"configurable": {"thread_id": session.id}})
    code = (snapshot.values or {}).get("code")
    if not code:
        raise HTTPException(status_code=400, detail="尚无生成结果")

    if req.accept:
        save_dir = Path(GENERATED_CODE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        await asyncio.to_thread(path.write_text, code, "utf-8")
        async with _lock:
            _sessions.pop(session.id, None)
        return {"type": "saved", "path": str(path)}

    feedback = (req.feedback or "").strip()
    if not feedback:
        async with _lock:
            _sessions.pop(session.id, None)
        return {"type": "ended", "message": "会话已结束, 代码未保存"}
    return {"type": "regenerate", "session_id": session.id}


@app.get("/api/health")
async def health():
    """健康检查: 进程活着 + 存储可达 + 当前 alias 指向哪个物理集合。"""
    store = app.state.store
    info = {"status": "ok", "sessions": len(_sessions)}
    try:
        info["collection"] = await asyncio.to_thread(store.resolve_alias)
        info["count"] = await asyncio.to_thread(store.count)
    except Exception as e:                      # 存储不可用不该让进程被判死
        info["status"] = "degraded"
        info["error"] = str(e)
    return info


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "index.html")


if __name__ == "__main__":
    # 直接传 app 对象, 不依赖模块导入字符串, PyCharm Run 与命令行均可
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False)
