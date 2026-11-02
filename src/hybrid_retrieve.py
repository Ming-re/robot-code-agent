"""两阶段检索的命令行脚本，调参用。

    召回：dense + sparse 混合检索，RRF 融合，取 --pool 条
    精排：候选池过一遍 ColBERT 词级重排，取 --top 条

池子要比最终条数大不少。RRF 只能重排已经召回的候选，不会创造召回；
两路各取 10 条、融合再取 10 条的话，正确答案排第 11 位就永远出不来。

线上 Agent 那一路只做了混合检索、没接重排，本地 CPU 逐条编码候选太慢。
这个脚本就是拿来量一下接上去能涨多少。

    python -m src.hybrid_retrieve "如何打开伺服"
    python -m src.hybrid_retrieve "PTP指令的格式" --pool 100 --top 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 项目根目录加入 sys.path, 保证以脚本方式（PyCharm Run）运行时也能导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_DIR,
    MILVUS_URI,
    RERANK_POOL_K,
)
from src.embedding.bge_m3 import BgeM3Encoder
from src.milvus_store import MilvusStore


def main():
    parser = argparse.ArgumentParser(
        description="两阶段检索: 混合召回(池) + ColBERT 重排(精)")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--pool", type=int, default=RERANK_POOL_K,
                        help="混合检索候选池大小（必须远大于 --top）")
    parser.add_argument("--top", type=int, default=3, help="最终返回条数")
    args = parser.parse_args()
    if args.pool < args.top:
        parser.error("--pool 必须 >= --top")

    encoder = BgeM3Encoder(EMBEDDING_MODEL_DIR)
    store = MilvusStore(MILVUS_URI, COLLECTION_NAME)

    print(f"查询: {args.query}")
    print(f"集合: {store.resolve_alias()}（别名 {COLLECTION_NAME}）\n")

    # 1. query 编码（这一层要 colbert，因为后面要重排）
    q_reps = encoder.encode_query(args.query, with_colbert=True)
    dense = q_reps["dense_vecs"][0].tolist()
    sparse = BgeM3Encoder.sparse_to_milvus(q_reps["lexical_weights"][0])
    print(f"query 编码: dense {len(dense)}d, sparse 词数 {len(sparse)}, "
          f"colbert {q_reps['colbert_vecs'][0].shape[0]} tokens")

    # 2. 召回: 混合检索 + RRF，只取候选池
    hits = store.hybrid_search(dense, sparse, top_k=args.pool)

    # 同一 chunk 可能因为历史重复入库出现多次，先按 chunk_id 去重
    unique, seen = [], set()
    for h in hits:
        cid = h["entity"]["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(h)

    print(f"\n--- 召回阶段: RRF 融合候选池 {len(unique)} 条 ---")
    for i, h in enumerate(unique[:10], 1):
        e = h["entity"]
        print(f"{i:>2}. [{e['chunk_id']}] {e['title']} (rrf={h['distance']:.5f})")
    if len(unique) > 10:
        print(f"    ... 其余 {len(unique) - 10} 条省略")

    # 3. 精排: ColBERT MaxSim
    texts = [h["entity"]["text"] for h in unique]
    p_reps = encoder.encode_for_rerank(texts)
    scores = encoder.colbert_score(q_reps, p_reps)
    order = np.argsort(-scores)

    print(f"\n--- 精排阶段: ColBERT MaxSim 重排 {len(unique)} 条 ---")
    for rank, idx in enumerate(order[: min(len(order), args.top + 2)], 1):
        e = unique[idx]["entity"]
        print(f"{rank:>2}. [{e['chunk_id']}] {e['title']} "
              f"(colbert={scores[idx]:.4f})")

    # 4. 最终结果
    print(f"\n===== 最终 TOP_{args.top} =====")
    for rank, idx in enumerate(order[: args.top], 1):
        e = unique[idx]["entity"]
        print(f"\n[{rank}] [{e['chunk_id']}] {e['title']} "
              f"(colbert={scores[idx]:.4f}, 页码 {e['pages']})")
        print(f"路径: {e['path']}")
        print(e["text"][:600])


if __name__ == "__main__":
    main()
