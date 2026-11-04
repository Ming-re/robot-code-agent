"""Agent 主图：agent 决策，tools 执行，ask 追问。

    START -> agent ─┬─ tools -> agent -> ...
                    ├─ ask   -> agent -> ...
                    └─ END

为什么把 ask 单独拆成一个节点：

interrupt() 恢复时，LangGraph 会把**整个节点从头重跑**（检查点只存节点输出，
不存"执行到哪一行"）。如果 ask 和检索待在同一个节点里，那么中断前的检索
在恢复后会被重新执行一遍——查库重来、生成重烧一遍 token。

以前是用 completed[tool_call_id] 记忆来绕的，但那个记忆本身是失效的：
它写回 state 的那一行在函数末尾，而 interrupt 抛异常时函数直接退出了，
那一行永远执行不到，所以它只在"不需要它的场景"里生效。

拆开之后就简单了：ask 节点里只调 interrupt，没有任何副作用，重放它不花代价。

chunks 仍然要挂 reducer：一个 superstep 里可能同时跑好几个检索，
整体替换的话后写会把先写盖掉，而且一点报错都没有。

emit 回调从 config["configurable"]["emit"] 取。
"""

import json
import os

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import Annotated, TypedDict

from ..config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_RETRIEVE_CALLS,
    RETRIEVAL_TOOLS,
)
from .reducer import merge_chunks
from .tools import TOOL_SCHEMAS, ToolExecutor

AGENT_SYSTEM_PROMPT = """你是工业机器人编程智能体，目标：根据用户需求生成完整可运行的机器人程序。

可用工具：
- retrieve_chunks(query_text, top_k): 在手册向量库中检索与查询文本相关的指令 chunk
- search_section(section_id, query_text?, top_k): 检索手册指定章节的指令（2.8.4=Modbus TCP、2.8.7=多网口通讯、
  2.8.1=IO、2.8.6=总线寄存器、2.3=运动、2.4=坐标系、2.5=位置姿态、2.7=流程、2.9.4=皮带、2.9.6=托盘）
- read_chunk(chunk_id): 读取指定 chunk 全文
- ask_user(questions): 需求信息不足时向用户追问（一次 3-5 个，不重复问已答过的；尽量单独调用）
- generate_code(requirement, chunk_ids): 用已检索的 chunk 生成程序代码（流式输出）
- check_instructions(): 自检当前代码的指令是否全部合法（含参数个数）
- fix_code(): 修正不合规指令（只允许等价替换）
- finish(summary): 交付最终代码

工作方式（自主决策）：
1. 先分析需求；信息不足（通讯参数/IO 映射/点位/异常处理等缺失且影响程序结构）时先 ask_user
2. 按需求涉及的指令族检索上下文：先 retrieve_chunks 按意图查，必要时 search_section 补全章节，
   拿不准的 chunk 用 read_chunk 细读确认
3. 上下文充分后 generate_code 生成
4. 生成后必须 check_instructions 自检；有违规就 fix_code，修正后再 check，直到通过
5. 全部通过后 finish 交付
规则：
- 指令必须来自手册或项目参考代码，绝不编造
- 与需求无关的检索结果不要采纳；检索一次不理想就换查询词再试
- 不要重复调用已执行过的相同工具（尤其不要死循环检索）
- 物理安全：自检不通过绝不交付；**绝不允许靠注释掉代码来通过自检**
- 用简洁中文与用户交流"""


class AgentState(TypedDict, total=False):
    """图状态。

    messages    由 add_messages 归约（追加）
    chunks      由 merge_chunks 归约（按 chunk_id 合并 + LRU 截断）
    pending_ask 有待处理的追问：{"call_id": ..., "questions": [...]}
    其余字段整体替换。
    """
    messages: Annotated[list, add_messages]
    requirement: str
    code: str
    chunks: Annotated[dict, merge_chunks]
    tokens: tuple          # (prompt_tokens, completion_tokens) 累计
    tool_count: int        # 检索类工具已经调用了几次（用于封顶）
    finished: bool
    pending_ask: dict


def _emit(config, event):
    """从图配置取 emit 回调推送轨迹事件（SSE 由 app 层注入）。"""
    emit = (config or {}).get("configurable", {}).get("emit")
    if emit:
        emit(event)


def _pending_ask_call(state):
    """从最后一条 AI 消息里找 ask_user 调用。"""
    last = state["messages"][-1]
    for tc in (getattr(last, "tool_calls", None) or []):
        if tc["name"] == "ask_user":
            return tc
    return None


def build_graph(executor: ToolExecutor):
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0.2,
        max_tokens=2048,
        timeout=180,
    ).bind_tools(TOOL_SCHEMAS)

    async def agent_node(state: AgentState):
        """决策节点: LLM 观察历史, 决定下一步（文字或工具调用）。

        顺便把这一轮的 token 累加进 state。决策阶段才是花钱的大头：
        每轮都要重读全部历史，十几轮下来 input 是 O(n²) 增长的。
        """
        resp = await llm.ainvoke(state["messages"])
        usage = getattr(resp, "usage_metadata", None) or {}
        prev_in, prev_out = state.get("tokens") or (0, 0)
        return {
            "messages": [resp],
            "tokens": (prev_in + int(usage.get("input_tokens", 0) or 0),
                       prev_out + int(usage.get("output_tokens", 0) or 0)),
        }

    async def tools_node(state: AgentState, config):
        """执行节点: 执行本批次里除 ask_user 以外的全部工具。

        这个节点里不会有 interrupt，所以它不会被重放——不需要任何记忆机制。
        """
        last = state["messages"][-1]
        calls = [tc for tc in last.tool_calls if tc["name"] != "ask_user"]

        local = dict(state)        # 同批次后续工具能看到前面工具产生的结果
        merged = {}
        new_chunks = {}
        tool_msgs = []
        used = int(state.get("tool_count") or 0)      # 检索类工具已经用了几次

        for tc in calls:
            call_id = tc["id"]
            _emit(config, {"type": "tool", "name": tc["name"],
                           "args": tc.get("args", {}), "call_id": call_id})

            # 检索次数封顶：超了就不执行，直接把"该生成了"塞回给模型。
            # 不封顶它会一直换查询词重试，token 全烧在无效检索上。
            if tc["name"] in RETRIEVAL_TOOLS and used >= MAX_RETRIEVE_CALLS:
                result, updates = {
                    "error": f"检索次数已达上限（{MAX_RETRIEVE_CALLS} 次）。"
                             "不要再检索，请立刻调用 generate_code 生成程序；"
                             "没确认的指令在 finish 时说明。"
                }, {}
            else:
                try:
                    result, updates, _ = await executor.execute(
                        local, tc["name"], tc.get("args", {}),
                        lambda evt: _emit(config, evt), call_id=call_id)
                except Exception as e:
                    result, updates = {"error": f"工具执行异常: {e}"}, {}
                if tc["name"] in RETRIEVAL_TOOLS:
                    used += 1

            tool_msgs.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=call_id))
            for key, value in (updates or {}).items():
                if key == "chunks":
                    new_chunks = merge_chunks(new_chunks, value)
                    local["chunks"] = merge_chunks(local.get("chunks"), value)
                else:
                    merged[key] = value
                    local[key] = value

        out = {"messages": tool_msgs}
        if new_chunks:
            out["chunks"] = new_chunks
        out["tool_count"] = used
        out.update(merged)

        # 每轮把累计用量推给前端/日志，钱花在哪要看得见
        tin, tout = state.get("tokens") or (0, 0)
        _emit(config, {"type": "usage", "prompt": tin,
                       "completion": tout, "tool_calls": used})

        ask_call = _pending_ask_call(state)
        if ask_call:
            out["pending_ask"] = {"call_id": ask_call["id"],
                                  "questions": ask_call["args"]["questions"]}
        return out

    async def ask_node(state: AgentState, config):
        """追问节点: 只做两件事——推事件、挂起等回答。

        这里故意不放任何副作用。interrupt 恢复时这个节点会从头重跑一遍，
        但重跑只意味着"再调一次 interrupt"，不会再查库、再调模型。
        """
        pending = state.get("pending_ask") or {}
        if not pending:
            call = _pending_ask_call(state)
            if call:
                pending = {"call_id": call["id"],
                           "questions": call["args"]["questions"]}

        questions = pending.get("questions") or []
        _emit(config, {"type": "ask", "questions": questions,
                       "call_id": pending.get("call_id")})

        answers = interrupt({"questions": questions})
        return {
            "messages": [ToolMessage(
                content=json.dumps({"status": "已收到用户回答", "answers": answers},
                                   ensure_ascii=False),
                tool_call_id=pending["call_id"])],
            "pending_ask": {},
        }

    def route_after_agent(state: AgentState):
        """有普通工具先去 tools；只有 ask 就单独走 ask；都没有就结束。"""
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return END
        if any(tc["name"] != "ask_user" for tc in calls):
            return "tools"
        return "ask"

    def route_after_tools(state: AgentState):
        if state.get("finished"):
            return END
        if state.get("pending_ask"):
            return "ask"
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("ask", ask_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent,
                                {"tools": "tools", "ask": "ask", END: END})
    graph.add_conditional_edges("tools", route_after_tools,
                                {"ask": "ask", "agent": "agent", END: END})
    graph.add_edge("ask", "agent")
    return graph.compile(checkpointer=InMemorySaver())
