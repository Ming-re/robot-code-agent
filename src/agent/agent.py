"""Agent CLI: 基于 LangGraph 的自主决策循环。

用法:
- PyCharm 中直接运行本文件（Run 按钮）: 无参数时交互式输入需求
- 或命令行:
    python -m src.agent.agent --demo
    python -m src.agent.agent "你的需求文本"
    python -m src.agent.agent --file data/demo_requirement.txt

ask_user 的 interrupt 暂停会打印问题并等待标准输入回答,
回答后从检查点恢复继续执行。
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录加入 sys.path, 保证以脚本方式（PyCharm Run）运行时也能导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from src.agent.generator import CodeGenerator
from src.agent.graph import AGENT_SYSTEM_PROMPT, build_graph
from src.agent.tools import ToolExecutor
from src.config import (
    CHUNKS_JSON,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL_DIR,
    GENERATED_CODE_DIR,
    MILVUS_URI,
)
from src.embedding.bge_m3 import BgeM3Encoder
from src.milvus_store import MilvusStore

DEMO_REQUIREMENT = DATA_DIR / "demo_requirement.txt"


def _emit_print(event):
    if event["type"] == "thought":
        print(f"\n[思考] {event['content']}")
    elif event["type"] == "tool":
        print(f"[工具] {event['name']} {json.dumps(event.get('args', {}), ensure_ascii=False)[:120]}")
    elif event["type"] == "ask":
        print(f"[追问] {event['questions']}")
    elif event["type"] == "delta":
        sys.stdout.write(event["text"])
        sys.stdout.flush()
    elif event["type"] == "fixed":
        print("\n[已自动修正不合规指令]")
    elif event["type"] == "done":
        print(f"\n[交付] {event.get('summary', '')}")
        print(f"[tokens] {event.get('tokens')}")


async def run_agent(requirement):
    generator = CodeGenerator()
    encoder = BgeM3Encoder(EMBEDDING_MODEL_DIR)
    store = MilvusStore(MILVUS_URI, COLLECTION_NAME)
    all_chunks = {c["chunk_id"]: c
                  for c in json.loads(Path(CHUNKS_JSON).read_text(
                      encoding="utf-8"))}
    executor = ToolExecutor(encoder, store, generator, all_chunks)
    graph = build_graph(executor)
    config = {"configurable": {"thread_id": "cli", "emit": _emit_print}}

    graph_input = {
        "requirement": requirement,
        "messages": [SystemMessage(content=AGENT_SYSTEM_PROMPT),
                     HumanMessage(content=requirement)],
    }
    print(f"需求({len(requirement)} 字) 提交给 Agent ...\n")
    while True:
        async for update in graph.astream(graph_input, config=config,
                                          stream_mode="updates"):
            for node, upd in update.items():
                if node != "agent" or "messages" not in upd:
                    continue
                # 思考文本在此打印; 工具事件由 tools_node 执行时推送
                for m in upd["messages"]:
                    if getattr(m, "content", ""):
                        _emit_print({"type": "thought",
                                     "content": m.content})
        snapshot = await graph.aget_state(config)
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            questions = snapshot.tasks[0].interrupts[0].value["questions"]
            print("\n" + "\n".join(f"  {i + 1}. {q}"
                                   for i, q in enumerate(questions)))
            answer = input("\n回答(每行一条, 对应上面的问题): ")
            graph_input = Command(resume=answer)
            continue
        break

    state = (await graph.aget_state(config)).values
    code = state.get("code")
    if code:
        path = (GENERATED_CODE_DIR /
                f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        print(f"\n代码已保存: {path}")
    else:
        print("\nAgent 未交付代码")


def main():
    parser = argparse.ArgumentParser(description="Agent CLI (LangGraph 自主循环)")
    parser.add_argument("requirement", nargs="?")
    parser.add_argument("--file", help="从文件读取需求")
    parser.add_argument("--demo", action="store_true",
                        help=f"使用示例需求 ({DEMO_REQUIREMENT.name})")
    args = parser.parse_args()

    if args.demo:
        args.file = str(DEMO_REQUIREMENT)
    if args.file:
        path = Path(args.file)
        if not path.exists():
            parser.error(f"找不到需求文件：{path}\n"
                         f"（--demo 用的是 {DEMO_REQUIREMENT.name}，仓库里没带，"
                         f"自己写一段放进去就行）")
        requirement = path.read_text(encoding="utf-8")
    elif args.requirement:
        requirement = args.requirement
    else:
        # PyCharm Run 无参数启动: 交互式输入需求（输入空行结束）
        print("请粘贴需求文本（输入空行结束）:")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        requirement = "\n".join(lines).strip()
        if not requirement:
            parser.error("需求为空")
    asyncio.run(run_agent(requirement))


if __name__ == "__main__":
    main()
