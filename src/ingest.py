"""把 chunks.json 灌进 Milvus。

    python -m src.ingest                       # 增量更新，只重算变了的 chunk
    python -m src.ingest --rebuild --promote   # 建新集合，校验通过后切别名
    python -m src.ingest --dry-run             # 只看看会改什么

写入是幂等的：chunk_id 做主键 + upsert，跑几遍库里都只有一份。
要发新版就新建一个带版本号的集合，切别名过去，旧集合留着当回滚点，
不用 drop，也不用停服。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录加入 sys.path, 保证以脚本方式（PyCharm Run）运行时也能导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CHUNKS_JSON,
    COLLECTION_NAME,
    COLLECTION_PREFIX,
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_DIR,
    MILVUS_URI,
)
from src.ingest_plan import build_row, collection_name, plan_updates
from src.milvus_store import MilvusStore


def main():
    parser = argparse.ArgumentParser(
        description="chunks 向量化并写入 Milvus（幂等 upsert）")
    parser.add_argument("--chunks", default=str(CHUNKS_JSON),
                        help="chunks.json 路径")
    parser.add_argument("--collection", default=None,
                        help="目标物理集合名；默认用当前别名指向的集合")
    parser.add_argument("--rebuild", action="store_true",
                        help="新建一个版本化集合做全量写入（不动线上集合）")
    parser.add_argument("--promote", action="store_true",
                        help="写入并校验通过后，把别名切到新集合")
    parser.add_argument("--version", default=None,
                        help="手册版本号（写入 doc_version，默认取时间戳）")
    parser.add_argument("--prune", action="store_true",
                        help="删除源数据里已不存在的 chunk")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要写入/删除的内容，不连接写库")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 个 chunk（调试用）")
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    if args.limit:
        chunks = chunks[: args.limit]
    doc_version = args.version or datetime.now().strftime("%Y%m%d_%H%M")
    print(f"读取 {len(chunks)} 个 chunk: {args.chunks}")

    store = MilvusStore(MILVUS_URI, COLLECTION_NAME)

    # ---- 决定目标集合 ----
    created_target = False
    if args.collection:
        target = args.collection
    elif args.rebuild:
        target = collection_name(COLLECTION_PREFIX, doc_version)
    else:
        target = store.resolve_alias()
        if not target:
            target = collection_name(COLLECTION_PREFIX, doc_version)
            created_target = True
            print(f"别名 {COLLECTION_NAME} 尚未建立，将新建: {target}")

    if not store.has_collection(target):
        store.create_collection(EMBEDDING_DIM, name=target)
        created_target = True
        print(f"已创建集合 {target} "
              f"(dense {EMBEDDING_DIM}d HNSW+COSINE, sparse 倒排+IP, "
              f"主键 chunk_id 非自增)")

    # ---- 差集: 只编码变化的 chunk ----
    existing = store.content_hash_map(target)
    plan = plan_updates(chunks, existing, prune=args.prune)
    print(f"库内 {len(existing)} 条；本次需写入 {len(plan['to_write'])} 条，"
          f"可复用 {len(plan['unchanged'])} 条，"
          f"待删除 {len(plan['to_delete'])} 条")

    if args.dry_run:
        for c in plan["to_write"][:10]:
            print(f"  + {c['chunk_id']} {c['title']}")
        if len(plan["to_write"]) > 10:
            print(f"  ... 共 {len(plan['to_write'])} 条待写")
        for cid in plan["to_delete"][:10]:
            print(f"  - {cid}")
        return

    if plan["to_write"]:
        from src.embedding.bge_m3 import BgeM3Encoder  # 延迟导入（需 torch）

        encoder = BgeM3Encoder(EMBEDDING_MODEL_DIR)
        print("向量化中 (dense + sparse) ...")
        out = encoder.encode_docs([c["text"] for c in plan["to_write"]],
                                  batch_size=args.batch_size)
        rows = [
            build_row(
                chunk,
                out["dense_vecs"][i].tolist(),
                BgeM3Encoder.sparse_to_milvus(out["lexical_weights"][i]),
                doc_version,
            )
            for i, chunk in enumerate(plan["to_write"])
        ]
        store.upsert(rows, name=target)
        print(f"upsert 完成: {len(rows)} 条")

    if plan["to_delete"]:
        store.delete_ids(plan["to_delete"], name=target)
        print(f"已删除 {len(plan['to_delete'])} 条过期 chunk")

    store.load(target)
    total = store.count(target)
    print(f"集合 {target} 现有 {total} 条记录")

    # ---- 校验后再切别名（fail-closed）----
    if created_target or args.promote:
        if total != len(chunks):
            raise SystemExit(
                f"校验失败: 期望 {len(chunks)} 条，实际 {total} 条；"
                f"已保留旧别名 {COLLECTION_NAME} 未切换")
        print(f"切换别名 {COLLECTION_NAME} -> {target}")
        print(f"当前别名指向: {store.promote(target)}")
    else:
        print(f"提示: 未切换别名，线上仍为 {store.resolve_alias()}；"
              f"确认无误后执行 --promote")


if __name__ == "__main__":
    main()
