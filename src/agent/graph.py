"""Agent 主图：agent 节点做决策，tools 节点执行，条件边路由。

    START -> agent -> tools -> agent -> ... -> END

两个坑记一下：

1. chunks 要用 reducer 合并。一个 superstep 里可能同时跑好几个检索，
   整体替换的话后写会把先写盖掉，而且一点报错都没有。
2. interrupt 恢复的时候，整个 tools 节点会从头再跑一遍。节点里用
   completed[tool_call_id] 记住执行过的工具，重放时直接拿结果，
   否则检索会重查、生成会重跑。

emit 回调从 config["configurable"]["emit"] 取。
"""

import json
import os

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from ..config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .reducer import merge_chunks, merge_dict
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

    messages  由 add_messages 归约（追加）
    chunks    由 merge_chunks 归约（按 chunk_id 合并 + LRU 截断，重放幂等）
    completed 由 merge_dict 归约（tool_call_id -> 上次执行结果，保证重放不重复执行）
    其余字段整体替换。
    """
    messages: Annotated[list, add_messages]
    requirement: str
    code: str
    chunks: Annotated[dict, merge_chunks]
    completed: Annotated[dict, merge_dict]
    tokens: tuple          # (prompt_tokens, completion_tokens)
    finished: bool


def _emit(config, event):
    """从图配置取 emit 回调推送轨迹事件（SSE 由 app 层注入）。"""
    emit = (config or {}).get("configurable", {}).get("emit")
    if emit:
        emit(event)


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
        """决策节点: LLM 观察历史, 决定下一步（文字或工具调用）。"""
        resp = await llm.ainvoke(state["messages"])
        return {"messages": [resp]}

    async def tools_node(state: AgentState, config):
        """执行节点: 执行上一条消息里的全部工具调用, 结果写回消息历史。

        - ask_user 排到最后：它会让整个节点重放，尽量缩小重放窗口；
        - 已执行过的 tool_call_id 直接复用结果（重放安全，不重复产生副作用）；
        - 同批次内后续工具能看到前面工具产生的上下文（local 状态）。
        """
        last = state["messages"][-1]
        calls = sorted(last.tool_calls,
                       key=lambda tc: 1 if tc["name"] == "ask_user" else 0)
        done = dict(state.get("completed") or {})
        local = dict(state)
        merged = {}
        tool_msgs = []

        for tc in calls:
            call_id = tc["id"]
            if call_id in done:                     # 节点重放: 复用上次结果
                result = done[call_id]["result"]
                updates = done[call_id]["updates"]
            else:
                _emit(config, {"type": "tool", "name": tc["name"],
                               "args": tc.get("args", {}), "call_id": call_id})
                try:
                    result, updates, _ = await executor.execute(
                        local, tc["name"], tc.get("args", {}),
                        lambda evt: _emit(config, evt), call_id=call_id)
                except GraphInterrupt:
                    raise  # ask_user 的暂停信号, 交给 LangGraph, 不能吞掉
                except Exception as e:
                    result, updates = {"error": f"工具执行异常: {e}"}, {}
                done[call_id] = {"result": result, "updates": updates}

            tool_msgs.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=call_id))
            for key, value in (updates or {}).items():
                if key == "chunks":
                    merged["chunks"] = merge_chunks(merged.get("chunks"), value)
                    local["chunks"] = merge_chunks(local.get("chunks"), value)
                else:
                    merged[key] = value
                    local[key] = value

        merged["messages"] = tool_msgs
        merged["completed"] = done
        return merged

    def route_after_agent(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def route_after_tools(state: AgentState):
        if state.get("finished"):
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent,
                                {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools,
                                {"agent": "agent", END: END})
    return graph.compile(checkpointer=InMemorySaver())
