# 检索评估（golden set）

这一层只测 **L1 检索**：给定一条 query，期望的 chunk 是否出现在前 k 名。

- 不需要调 LLM → 秒级跑完、零 token 成本 → **每次改检索参数都应该跑一遍**
- 输出 Recall@1/5/10、MRR、coverage@5（多意图）、负例拒答率，并与 baseline 做 diff

## 快速开始

```bash
# 1. 先确保 Milvus 起了、数据入库了
docker compose up -d
python -m src.ingest

# 2. 跑评估（会打印指标 + 与 baseline 的 diff）
python -m eval.run_retrieval

# 3. 确认变好之后，把这版写成新基线
python -m eval.run_retrieval --update-baseline
```

报告落在 `eval/reports/report_<时间戳>.json`，基线在 `eval/baseline.json`。

## 指标定义

| 指标 | 定义 | 回答什么 |
|---|---|---|
| `recall@k` | 期望 chunk 出现在前 k 名的 query 占比 | 有没有**找到** |
| `mrr` | `mean(1 / 首个命中名次)` | 排得**多高** |
| `coverage@5` | `|前5名 ∩ 期望集合| / |期望集合|` 的均值 | 多意图需求**覆盖了多少** |
| `negative.abstain_rate` | 负例中 `top1 RRF 分数 < floor` 的比例 | 会不会**硬凑一条** |

`floor` 默认 0.02（`config.RRF_ANSWER_FLOOR`）。这个阈值来自 RRF 的分数结构：
两路各取 `pool` 条、`k=60` 时，**单路命中**的分数上限约为 `1/61 ≈ 0.0164`，
**双路共识**才会超过 `0.03`。所以 0.02 大致是"两路都说有"的分界，可用 `--pool` 变化后重标。

## 为什么不用 top-1 准确率

1. 一个指令族常有多个相关 chunk，top-1 会把"排第 2 但没错"也判错；
2. 你的链路是把 top-k 喂给 LLM，真正决定生成质量的是 `recall@5`；
3. top-1 无法反映"排第 1 但分数极低"的假命中。

## 评估集怎么扩（当前 24 条）

`queries.jsonl` 每行一条：

```json
{"id": "q004", "query": "读 PLC 寄存器里的值",
 "expected_titles": ["ReadPortRegister 函数"], "type": "single", "notes": ""}
```

- `expected_titles` 用**标题**而不是 chunk_id：标题改了也不影响评估集；
  脚本会自己映射成 chunk_id，映射不到会打印 `[warn]`（这就是"标注写错了"的告警）。
- `type`：`single` / `paraphrase` / `multi` / `negative`，报告按类型分组出指标——
  真实瓶颈常常只在某一类上，混在一起会被平均掉。

### 三条标注铁律

1. **query 用手业务语言写，不要出现手册函数名。**
   出现函数名的 query 测的是关键词匹配，不是语义召回，会把指标虚高。
2. **必须有负例。** 只测正例的评估集永远偏高；负例才能测出"幻觉式检索"。
3. **不要用同一个模型既生成 query 又当 judge。**（当前这 24 条是人工挑的，没有这个问题。）

## 已知局限（写在代码里，也写在文档里）

- **n=24 很小**：只能证明"单意图检索基本可用"，不足以支撑"检索精度 xx%"这类泛化结论。
  扩到 100+ 之前，任何百分比都应标注样本量。
- **只覆盖 L1**：Agent 还有一层"LLM 决定检索什么"的决策，需要轨迹级样本才能评估，未做。
- **生成质量未评估**：L2 里可确定性计算的指标（词法合规、参数元数、需求覆盖度）已部分实现
  在 `src/agent/validator.py`，但还没有对应的 golden set。
- **跨版本库不一样就没法比**：`baseline.json` 里记了 collection 名，换索引要重建基线。

## 下一步（按性价比排序）

1. 把 n 从 24 扩到 60：优先补 `paraphrase`（词表错配是当前主要失败模式）。
2. 给 `multi` 类补 5 条真实需求（从 `data/demo_requirement.txt` 拆句子）。
3. L2 评估：把需求拆成"动作清单"，计算**需求覆盖度**——这条指标无法通过注释代码提高，
   正好用来兜住"修复轮把代码删掉"的漏洞。
4. 挂 CI：`run_retrieval` 只跑 L1 很快，可以做成"recall@5 不许低于基线"的门禁。
