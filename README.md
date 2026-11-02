# robot-code-agent

给工业机器人写程序时，指令手册一百多页，光查函数就够呛。这个项目做的是：
你用中文描述一段动作流程，它自己去手册里翻对应的指令，缺信息会反问你，
然后把程序写出来，最后过一层校验才交给你。

目前只在一台机器人的私有指令语言（`.prg`）上跑通了，换语言要改切块规则和校验白名单。

---

## 整体是怎么走的

```
PDF 手册 ──切块──> chunks.json ──bge-m3──> Milvus ──┐
                                                    ├─> LangGraph Agent ─> 程序代码 ─> 校验
需求（中文）────────────────────────────────────────┘
```

检索这块是 dense + sparse 两路召回、RRF 融合，候选池再可选用 ColBERT 重排。
Agent 那边 8 个工具，检索哪些章节、要不要追问、自检没过怎么修，都是模型自己决定。

---

## 跑起来

仓库里带了一份虚构的示例知识库（`examples/sample_chunks.json`，11 条编造的指令），
不用自己的手册也能先把流程跑通：

```bash
docker compose up -d                          # Milvus
python -m src.ingest --chunks examples/sample_chunks.json --collection demo_chunks
python -m unittest discover -s tests -t .
python -m src.agent.app                       # http://127.0.0.1:8000
```

换成自己的手册：

```bash
# 手册放 data/raw/manual.pdf（或者设 ROBOT_DATA_DIR 指到项目外面）
python -m src.chunker.pdf_chunker   # 切块 -> data/chunks/
python -m src.verify_chunks         # 编号连续性、正文非空、围栏配平
python -m src.ingest                # 入库

# 命令行也可以直接跑
python -m src.agent.agent "把 A 点的物料搬到 B 点，到位后给 PLC 一个完成信号"
```

需要 `DEEPSEEK_API_KEY`。

关于数据目录：默认是项目下的 `./data`。如果你不想让手册和切块产物躺在仓库里，
设一个环境变量就行：

```bash
set ROBOT_DATA_DIR=D:\robot-data      # Windows
export ROBOT_DATA_DIR=/data/robot     # Linux / macOS
```

单测里有 5 个要用到切块产物和 `main.prg`，这两个文件不在仓库里，缺的时候自动跳过。

`--demo` 读的是 `data/demo_requirement.txt`，仓库里也没带（那是我自己机器上的工艺需求）。
自己写一段放进去就能用。

依赖 Python 3.11。模型（bge-m3）自己去 HuggingFace 下，放 `models/embedding/bge-m3/`，
体积太大没往仓库里放。

`main.prg` 是注入给模型的「项目参考代码」，用来对齐命名和写法。
仓库里没带（我手上那份不方便公开），你放自己的一份进去效果会更好；
没有也能跑，只是生成出来的风格会比较随意。

---

## 目录

```
src/chunker/           PDF 切块
src/embedding/         bge-m3 三路编码
src/milvus_store.py    建表 / 写入 / 检索 / 别名
src/ingest.py          入库（幂等 upsert，支持蓝绿发布）
src/ops.py             别名运维：status / promote / rollback
src/agent/             graph.py 是图，tools.py 是工具，validator.py 是指令校验
eval/                  检索评估，24 条 query
tests/                 单测
```

---

## 踩过的坑

- **Milvus 的 schema 改不了**。想加字段只能新建集合并把数据重灌一遍，
  所以索引这块一开始就该用别名（应用查别名，物理集合带版本号），
  不然每次改字段都得停服。
- **pymilvus 升大版本后参数名变过**。`hybrid_search` 是 `ranker` 不是 `rerank`，
  `search` 要显式给 `anns_field`。装的时候按 `requirements.txt` 的下限来。
- **LangGraph 的 interrupt 恢复会把整个节点从头重跑**，不是从中断那一行接着走。
  所以 ask_user 之前做过的事（检索、生成、推送事件）会再来一遍。
  我用 `completed[tool_call_id]` 记已执行过的工具来绕开，代价是检查点变大。
- **FlagEmbedding 的 `colbert_score` 一次只吃一对**（单查询对单文档），
  候选多的时候得自己写循环逐对算。
- **Milvus 刚写完马上 count 可能是 0**，Bounded 一致性下有可见性延迟，得 load 再重试。

---

## 还没做 / 已知问题

- 检索评估只标了 24 条 query，样本太小，只能说明单意图检索还行。
- 线上那一路只做了混合召回，没接 ColBERT 重排。本地 CPU 逐条编码候选太慢，
  真要用得先把文档侧的 ColBERT 向量预计算存起来。
- 会话状态在内存里（`InMemorySaver`），重启就没了，也没法多实例。
- 生成质量没有端到端评估，现在能自动算的只有词法合规和参数个数。
- 校验只管词法、参数个数和安全动作有没有被删。
  「能不能跑」和「逻辑对不对」它管不了。

---

## License

MIT

