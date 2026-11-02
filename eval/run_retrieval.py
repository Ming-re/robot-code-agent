"""跑一遍 golden set，出 recall@k / mrr / coverage，顺便和基线对比。

    python -m eval.run_retrieval
    python -m eval.run_retrieval --update-baseline

只测检索层，不调 LLM，几秒跑完，改完检索参数随手跑一下就行。
生成质量那层还没做。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import aggregate, diff
from src.config import (
    CHUNKS_JSON,
    COLLECTION_NAME,
    EMBEDDING_MODEL_DIR,
    EVAL_BASELINE,
    EVAL_QUERIES,
    EVAL_REPORT_DIR,
    MILVUS_URI,
    RRF_ANSWER_FLOOR,
)


def load_queries(path):
    queries = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8")
                                    .splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            queries.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{line_no} JSON 解析失败: {e}")
    return queries


def resolve_expected(chunks, titles):
    """标题 -> chunk_id（大小写与空白不敏感），返回 (ids, 未匹配到的标题)。"""
    index = {}
    for c in chunks:
        index[c["title"].strip().lower()] = c["chunk_id"]
    ids, missing = [], []
    for title in titles:
        cid = index.get(title.strip().lower())
        if cid is None:
            missing.append(title)
        else:
            ids.append(cid)
    return ids, missing


def main():
    parser = argparse.ArgumentParser(description="检索评估（Recall@k / MRR）")
    parser.add_argument("--queries", default=str(EVAL_QUERIES))
    parser.add_argument("--chunks", default=str(CHUNKS_JSON))
    parser.add_argument("--pool", type=int, default=50,
                        help="融合候选池大小（必须远大于 top-k）")
    parser.add_argument("--top-k", type=int, default=10,
                        help="按前 k 名计算 recall@k 的上限")
    parser.add_argument("--update-baseline", action="store_true",
                        help="把本次结果写成新的 baseline")
    parser.add_argument("--no-write", action="store_true", help="不落盘报告")
    args = parser.parse_args()

    from src.embedding.bge_m3 import BgeM3Encoder
    from src.milvus_store import MilvusStore

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    queries = load_queries(args.queries)
    print(f"加载 {len(queries)} 条 query, {len(chunks)} 个 chunk")

    encoder = BgeM3Encoder(EMBEDDING_MODEL_DIR)
    store = MilvusStore(MILVUS_URI, COLLECTION_NAME)
    print(f"检索集合: {store.resolve_alias()}（别名 {COLLECTION_NAME}）")

    rows, problems = [], []
    for q in queries:
        expected, missing = resolve_expected(chunks, q.get("expected_titles", []))
        if missing:
            problems.append(f"{q['id']}: 标题未找到 {missing}")
        out = encoder.encode_query(q["query"])
        dense = out["dense_vecs"][0].tolist()
        sparse = BgeM3Encoder.sparse_to_milvus(out["lexical_weights"][0])
        hits = store.hybrid_search(dense, sparse, top_k=args.pool)

        ranked, seen = [], set()
        for hit in hits:                       # 去重：重复入库会让同一 chunk 占多个名次
            cid = hit["entity"]["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            ranked.append(cid)
        top_score = float(hits[0]["distance"]) if hits else 0.0
        rows.append({
            "id": q["id"],
            "type": q.get("type", "unknown"),
            "query": q["query"],
            "expected": expected,
            "ranked": ranked[:max(args.top_k, 10)],
            "top_score": top_score,
        })

    report = aggregate(rows, ks=(1, 5, 10), floor=RRF_ANSWER_FLOOR)
    report["meta"] = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "n": len(queries),
        "pool": args.pool,
        "collection": store.resolve_alias(),
        "queries": str(args.queries),
    }
    if problems:
        report["problems"] = problems
        for p in problems:
            print(f"[warn] {p}")

    # ---- 打印 ----
    def show(title, block):
        if not block:
            return
        print(f"\n{title}")
        for key in ("n", "recall@1", "recall@5", "recall@10", "mrr",
                    "coverage@5", "miss"):
            if key in block:
                print(f"  {key:<12}{block[key]}")

    show("整体", report["overall"])
    for type_name, block in report["by_type"].items():
        show(f"类型: {type_name}", block)
    if "negative" in report:
        print(f"\n负例（n={report['negative']['n']}）"
              f"拒答率 {report['negative']['abstain_rate']}"
              f"（阈值 top1 RRF < {report['negative']['floor']}）")

    # ---- 与 baseline 对比（关键：没有 diff 的评估等于没有评估）----
    if Path(EVAL_BASELINE).exists():
        baseline = json.loads(Path(EVAL_BASELINE).read_text(encoding="utf-8"))
        print("\n相对 baseline:")
        for key, before, after, delta in diff(baseline, report):
            flag = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"  {key:<12}{before} -> {after}  {flag}{abs(delta)}")
    else:
        print("\n（没有 baseline，本次结果可用 --update-baseline 作为基线）")

    misses = [d for d in report["details"] if d["first_hit_rank"] is None]
    if misses:
        print(f"\n完全没命中的 query（{len(misses)} 条，需逐条看是否标注错了）:")
        for d in misses[:10]:
            print(f"  {d['id']} expected={d['expected']}")

    # ---- 落盘 ----
    if not args.no_write:
        EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EVAL_REPORT_DIR / (
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\n报告: {out_path}")
    if args.update_baseline:
        Path(EVAL_BASELINE).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已更新 baseline: {EVAL_BASELINE}")


if __name__ == "__main__":
    main()
