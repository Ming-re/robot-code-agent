"""8 个工具的定义和执行器。

工具由 graph 里的 tools 节点执行，状态从 AgentState 传进来。
ask_user 不在这里执行——它由 graph.py 的 ask 节点处理（那边只调 interrupt，
没有副作用，所以 interrupt 恢复重放时不会有任何代价）。

工具结果是给模型看的，chunk 只放预览，要全文让它自己 read_chunk 取，省 token。
"""

import asyncio

from ..embedding.bge_m3 import BgeM3Encoder

# 工具结果里 chunk 文本的截断长度（越小越省 token，全文按需 read_chunk）
_CHUNK_PREVIEW = 200
_FULL_TEXT_LIMIT = 3000
_MAX_TOP_K = 8

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "retrieve_chunks",
        "description": "在手册向量库中混合检索与查询文本相关的指令 chunk"
                      "（dense+sparse）。查询文本应聚焦单一意图。",
        "parameters": {"type": "object", "properties": {
            "query_text": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        }, "required": ["query_text"]}}},
    {"type": "function", "function": {
        "name": "search_section",
        "description": "检索手册指定章节里的指令（如 2.8.4=Modbus TCP、"
                      "2.8.7=多网口通讯、2.8.1=IO、2.3=运动、2.9.4=皮带、2.9.6=托盘）。"
                      "给 query_text 时在该章节内做语义排序；不给则按章节顺序列出。",
        "parameters": {"type": "object", "properties": {
            "section_id": {"type": "string"},
            "top_k": {"type": "integer", "default": 6},
            "query_text": {"type": "string",
                           "description": "可选的章节内语义排序文本"},
        }, "required": ["section_id"]}}},
    {"type": "function", "function": {
        "name": "read_chunk",
        "description": "读取指定 chunk 的完整文本（先检索拿到 chunk_id 再按需细读）。",
        "parameters": {"type": "object", "properties": {
            "chunk_id": {"type": "string"},
        }, "required": ["chunk_id"]}}},
    {"type": "function", "function": {
        "name": "ask_user",
        "description": "需求信息不足时向用户追问。只追问缺失且影响程序结构的关键信息,"
                      "一次 3-5 个问题, 不要重复问已经回答过的。调用后等待用户回答。"
                      "本工具应尽量单独调用（同一批里只放它一个）。",
        "parameters": {"type": "object", "properties": {
            "questions": {"type": "array", "items": {"type": "string"}},
        }, "required": ["questions"]}}},
    {"type": "function", "function": {
        "name": "generate_code",
        "description": "用已收集的 chunk 上下文生成完整机器人程序代码。"
                      "chunk_ids 为之前 retrieve_chunks/search_section 返回的编号;"
                      "留空则用已收集的全部 chunk。生成过程会流式输出。",
        "parameters": {"type": "object", "properties": {
            "requirement": {"type": "string"},
            "chunk_ids": {"type": "array", "items": {"type": "string"}},
        }, "required": ["requirement"]}}},
    {"type": "function", "function": {
        "name": "check_instructions",
        "description": "自检当前代码: 每条指令调用是否在手册/项目代码白名单内, "
                      "参数个数是否符合手册『格式』。返回违规清单（名称+行号+原因）。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "fix_code",
        "description": "对当前代码中的不合规指令做一轮 LLM 自动修正"
                      "（只允许等价替换，禁止注释掉代码）。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "任务完成, 交付最终代码。交付前必须已通过 check_instructions。",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "交付说明（简述覆盖的功能点）"},
        }, "required": ["summary"]}}},
]


class ToolExecutor:
    """执行工具调用。

    execute(state, name, args, emit, call_id) ->
        (结果文本, 状态更新 dict, 是否暂停)
    emit(event): 向用户前端推送轨迹事件（工具执行过程可见）。
    call_id: LLM 生成的工具调用 id，用于服务端对重放产生的事件去重。
    """

    def __init__(self, encoder, store, generator, all_chunks):
        self._encoder = encoder
        self._store = store
        self._generator = generator
        self._all_chunks = all_chunks  # chunk_id -> chunk dict（读全文用）

    async def execute(self, state, name, args, emit, call_id=None):
        """执行工具。"""
        if name == "retrieve_chunks":
            result, updates = await self._retrieve(args)
            return result, updates, False
        if name == "search_section":
            result, updates = await self._search_section(args)
            return result, updates, False
        if name == "read_chunk":
            return self._read_chunk(args), {}, False
        if name == "ask_user":
            # ask_user 由 graph.py 的 ask 节点处理：那个节点里只调 interrupt，
            # 没有任何副作用，所以重放不花代价。
            # 走到这里说明路由写错了，直接炸掉，不要静默兜底。
            raise RuntimeError("ask_user 不应该进工具执行器，请检查 graph 的路由")
        if name == "generate_code":
            result, updates = await self._generate(state, args, emit)
            return result, updates, False
        if name == "check_instructions":
            return self._check(state, emit), {}, False
        if name == "fix_code":
            result, updates = await self._fix(state, emit)
            return result, updates, False
        if name == "finish":
            code = state.get("code", "") or ""
            if not code:
                return {"error": "尚无代码可交付, 请先 generate_code"}, {}, False
            tokens = state.get("tokens") or (0, 0)
            emit({"type": "done", "code": code,
                  "summary": args.get("summary", ""),
                  "tokens": {"prompt": tokens[0], "completion": tokens[1]}})
            return {"status": "已交付"}, {"finished": True}, False
        return {"error": f"未知工具: {name}"}, {}, False

    # ---------- 检索工具 ----------

    @staticmethod
    def _hit_items(hits, new_chunks):
        items = []
        for h in hits:
            e = h["entity"] if isinstance(h, dict) and "entity" in h else h
            new_chunks[e["chunk_id"]] = e
            items.append({"chunk_id": e["chunk_id"], "title": e["title"],
                          "preview": e["text"][:_CHUNK_PREVIEW]})
        return items

    async def _retrieve(self, args):
        query = args["query_text"]
        top_k = min(int(args.get("top_k", 5)), _MAX_TOP_K)
        out = await asyncio.to_thread(self._encoder.encode_query, query,
                                      with_colbert=False)
        dense = out["dense_vecs"][0].tolist()
        sparse = BgeM3Encoder.sparse_to_milvus(out["lexical_weights"][0])
        hits = await asyncio.to_thread(
            self._store.hybrid_search, dense, sparse, top_k)
        new_chunks = {}
        items = self._hit_items(hits, new_chunks)
        # 只回传增量, 由 merge_chunks 合并（同一 superstep 多次检索不会互相覆盖）
        return ({"query": query, "results": items,
                 "hint": "这些 chunk 已记入上下文; 需要全文用 read_chunk"},
                {"chunks": new_chunks})

    async def _search_section(self, args):
        section = args["section_id"]
        top_k = min(int(args.get("top_k", 6)), _MAX_TOP_K + 2)
        query = (args.get("query_text") or "").strip()
        new_chunks = {}
        if query:
            # 有真实语义查询: 章节过滤 + 向量排序
            out = await asyncio.to_thread(self._encoder.encode_query, query,
                                      with_colbert=False)
            dense = out["dense_vecs"][0].tolist()
            hits = await asyncio.to_thread(
                self._store.search_filtered, dense, section, top_k)
        else:
            # 无查询文本: 纯结构查询（按 chunk_id 列出章节内容）。
            # 别拿 "2.8.4 章节" 这种没有语义的串去编码，排出来的顺序等于随机。
            hits = await asyncio.to_thread(
                self._store.query_by_prefix, section, top_k)
        items = self._hit_items(hits, new_chunks)
        return {"section": section, "results": items}, {"chunks": new_chunks}

    def _read_chunk(self, args):
        cid = args["chunk_id"]
        c = self._all_chunks.get(cid)
        if not c:
            return {"error": f"找不到 chunk {cid}"}
        return {"chunk_id": cid, "title": c["title"],
                "text": c["text"][:_FULL_TEXT_LIMIT]}

    # ---------- 生成/校验/修复 ----------

    async def _generate(self, state, args, emit):
        requirement = args.get("requirement") or state.get("requirement", "")
        known = state.get("chunks") or {}
        cids = args.get("chunk_ids") or list(known)
        chunks = [self._as_chunk(known[cid]) for cid in cids if cid in known]
        if not chunks:
            return ({"error": "尚未检索到任何指令上下文, "
                              "请先调用 retrieve_chunks 或 search_section"}, {})
        parts = []
        tokens = (0, 0)
        try:
            async for kind, payload in self._generator.stream_generate(
                    requirement, chunks):
                if kind == "delta":
                    parts.append(payload)
                    emit({"type": "delta", "text": payload})
                else:
                    tokens = payload
        except Exception as e:
            return {"error": f"生成失败: {e}"}, {}
        code = "".join(parts)
        prev_in, prev_out = state.get("tokens") or (0, 0)
        total = (prev_in + tokens[0], prev_out + tokens[1])
        return ({"status": "生成完成", "code_length": len(code),
                 "tokens": {"prompt": total[0], "completion": total[1]}},
                {"code": code, "tokens": total})

    @staticmethod
    def _as_chunk(entity):
        return {"chunk_id": entity["chunk_id"], "title": entity["title"],
                "text": entity["text"]}

    def _check(self, state, emit=None):
        code = state.get("code", "") or ""
        if not code:
            return {"error": "尚无代码可自检, 请先 generate_code"}
        report = self._generator.check_report(code)
        errors = [v for v in report if v["severity"] == "error"]
        warnings = [v for v in report if v["severity"] == "warning"]
        if emit and warnings:
            emit({"type": "warn", "items": [
                {"line": v["line"], "name": v["name"], "message": v["message"]}
                for v in warnings[:10]]})
        if not errors:
            return {"valid": True,
                    "message": "全部指令词法合规（不等于可运行）",
                    "warnings": [{"line": v["line"], "name": v["name"],
                                  "message": v["message"]}
                                 for v in warnings[:10]]}
        return {"valid": False,
                "violations": [{"name": v["name"], "line": v["line"],
                                "kind": v["kind"],
                                "message": v["message"],
                                "text": v["text"][:80]} for v in errors[:20]],
                "hint": "调用 fix_code 做等价替换修正；不要靠注释掉代码通过自检"}

    async def _fix(self, state, emit):
        code = state.get("code", "") or ""
        if not code:
            return {"error": "尚无代码可修正"}, {}
        try:
            fixed, changed = await self._generator.verify_and_fix(code)
        except Exception as e:
            return {"error": f"修正失败: {e}"}, {}
        if not changed:
            return {"status": "无需修正",
                    "violations_after": len(
                        self._generator.check_instructions(code))}, {}
        # fail-closed: 修复轮不允许以"删掉安全动作"为代价通过自检
        dropped = self._generator.safety_regressions(code, fixed)
        if dropped:
            return {"error": "修复被拒绝: 修正稿删除了安全动作",
                    "dropped_safety": dropped,
                    "hint": "必须用等价指令替换；安全动作禁止删除或注释"}, {}
        emit({"type": "fixed", "code": fixed})
        remaining = self._generator.check_instructions(fixed)
        unresolved = self._generator.unresolved_markers(fixed)
        return ({"status": "已修正" if not remaining else "仍不合规",
                 "violations_after": len(remaining),
                 "unresolved": unresolved,
                 "hint": "若仍有违规, 请重新 generate_code 补全上下文, "
                         "不要继续注释代码"},
                {"code": fixed})
