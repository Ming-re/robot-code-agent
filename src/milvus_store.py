"""Milvus 的建表、写入和检索封装。

主键用 chunk_id，非自增。自增的话同一份数据能插出无数条，检索时同一个内容
占掉两个候选位，后面的真候选就挤不进来。

应用层只认别名，物理集合带版本号，切别名就是发布，切回来就是回滚。
Milvus 的 schema 改不了，要加字段只能新建集合，正好也是同一套流程。
"""

import re
import time

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

_OUTPUT_FIELDS = ["chunk_id", "title", "level", "path", "pages", "text"]
_SECTION_RE = re.compile(r"^[0-9A-Za-z_.\-]{1,32}$")


def _safe_section(section):
    """校验章节号，避免把用户输入直接拼进 filter 表达式。"""
    section = (section or "").strip()
    if not _SECTION_RE.match(section):
        raise ValueError(f"非法的章节号: {section!r}")
    return section


def _chunk_sort_key(row):
    """按 chunk_id 的数值序排序（'2.10.1' 要排在 '2.9.1' 之后）。"""
    parts = []
    for piece in str(row.get("chunk_id", "")).split("."):
        digits = re.sub(r"\D", "", piece)
        parts.append((0, int(digits)) if digits else (1, 0))
    return parts


class MilvusStore:
    def __init__(self, uri, collection_name):
        self.client = MilvusClient(uri=uri)
        # collection_name 是别名（逻辑名），物理集合名走 resolve_alias()
        self.collection_name = collection_name

    # ------------------------------------------------------------ 集合与别名

    def has_collection(self, name=None):
        return self.client.has_collection(name or self.collection_name)

    def list_physical(self):
        """列出所有物理集合（排除别名本身）。"""
        try:
            return sorted(c for c in self.client.list_collections()
                          if c != self.collection_name)
        except Exception:
            return []

    def resolve_alias(self, alias=None):
        """别名 -> 物理集合名；如果没有别名，返回同名物理集合；都不存在返回 None。"""
        alias = alias or self.collection_name
        try:
            info = self.client.describe_alias(alias)
        except Exception:
            return alias if self.client.has_collection(alias) else None
        if isinstance(info, dict):
            return info.get("collection_name") or info.get("collection") or alias
        return getattr(info, "collection_name", alias)

    def promote(self, physical, alias=None):
        """把别名原子指向物理集合（蓝绿切换 / 回滚都走这里）。"""
        alias = alias or self.collection_name
        if not self.client.has_collection(physical):
            raise ValueError(f"物理集合不存在: {physical}")
        try:
            self.client.alter_alias(collection_name=physical, alias=alias)
        except Exception:
            # 别名还不存在时 alter_alias 会失败，改用 create
            self.client.create_alias(collection_name=physical, alias=alias)
        return self.resolve_alias(alias)

    def create_collection(self, dim, name=None, drop_if_exists=False):
        """建集合。默认不删已存在的同名集合。"""
        target = name or self.collection_name
        if self.client.has_collection(target):
            if not drop_if_exists:
                return target
            self.client.drop_collection(target)

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True,
                         max_length=64)
        schema.add_field("title", DataType.VARCHAR, max_length=256)
        schema.add_field("level", DataType.INT64)
        schema.add_field("path", DataType.VARCHAR, max_length=1024)
        schema.add_field("pages", DataType.VARCHAR, max_length=32)
        schema.add_field("char_count", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_version", DataType.VARCHAR, max_length=32)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        self.client.create_collection(
            target, schema=schema, index_params=index_params
        )
        return target

    def drop_collection(self, name):
        self.client.drop_collection(name)

    # ------------------------------------------------------------ 写入

    def upsert(self, rows, name=None):
        """按主键 upsert：重复跑 N 次结果一致（幂等）。"""
        return self.client.upsert(name or self.collection_name, data=rows)

    def insert(self, rows, name=None):
        """纯插入（会产生重复记录），仅保留做对照，入库请用 ``upsert``。"""
        return self.client.insert(name or self.collection_name, data=rows)

    def delete_ids(self, ids, name=None):
        if not ids:
            return None
        return self.client.delete(name or self.collection_name, ids=list(ids))

    def load(self, name=None):
        self.client.load_collection(name or self.collection_name)

    def flush(self, name=None):
        """把内存里的数据落盘并触发建索引。

        少了这一步，upsert 之后立刻查会拿到 0 条：Bounded 一致性下数据还没可见，
        索引也还在后台建。187 条实测要等十几秒才查得到，flush 之后是立刻可见。
        """
        return self.client.flush(name or self.collection_name)

    def count(self, name=None):
        """集合行数。先 flush 再 load，然后轮询到能查到为止。"""
        target = name or self.collection_name
        try:
            self.client.flush(target)
            self.client.load_collection(target)
        except Exception:
            pass
        for _ in range(10):
            try:
                result = self.client.query(
                    target, filter="", output_fields=["count(*)"])
                if result and result[0]["count(*)"]:
                    return result[0]["count(*)"]
            except Exception:
                pass
            try:
                rows = self.client.get_collection_stats(target).get("row_count", 0)
                if rows:
                    return rows
            except Exception:
                pass
            time.sleep(3)
        return 0

    def content_hash_map(self, name=None):
        """chunk_id -> content_hash，用于增量更新时判断"哪些 chunk 变了"。"""
        target = name or self.collection_name
        rows = []
        for expr in ("", 'chunk_id like "%"'):
            try:
                rows = self.client.query(
                    target, filter=expr,
                    output_fields=["chunk_id", "content_hash"],
                    limit=16384)
                break
            except Exception:
                continue
        return {r["chunk_id"]: r.get("content_hash", "") for r in rows}

    # ------------------------------------------------------------ 检索

    def query_by_prefix(self, section, limit=6, name=None):
        """按 chunk_id 前缀列出某章节的 chunk，完全不走向量。

        以前是拿 "2.8.4 章节" 这种串去编码再在子集里排序，那串文本没有语义，
        排出来基本是随机的；真正有用的只有前缀过滤。
        """
        sec = _safe_section(section)
        rows = self.client.query(
            name or self.collection_name,
            filter=f'chunk_id like "{sec}%"',
            output_fields=_OUTPUT_FIELDS,
            limit=max(int(limit), 100),
        )
        rows.sort(key=_chunk_sort_key)
        return rows[:int(limit)]

    def search_filtered(self, query_vec, section, top_k=6, name=None):
        """章节过滤 + dense 语义排序（Agent 目录补充用）。"""
        sec = _safe_section(section)
        return self.client.search(
            name or self.collection_name,
            data=[query_vec],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": 128}},
            filter=f'chunk_id like "{sec}%"',
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
        )[0]

    def hybrid_search(self, dense_vec, sparse_weights, top_k=10, name=None):
        """混合检索: dense ANN + sparse 倒排，RRF 融合后返回 top_k。

        dense_vec: 归一化 1024 维 list; sparse_weights: {token_id: weight} dict

        这里的 limit 是候选池大小，要远大于最终返回条数。两路各取 10 条、
        融合再取 10 条的话池子只有 20 条，正确答案排第 11 位就没了。
        RRF 只能重排已召回的候选，不会创造召回。
        """
        pool = max(int(top_k), 1)
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=pool,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_weights],
            anns_field="sparse",
            param={"metric_type": "IP"},
            limit=pool,
        )
        hits = self.client.hybrid_search(
            name or self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=pool,
            output_fields=_OUTPUT_FIELDS,
        )
        return hits[0]
