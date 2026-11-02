"""入库差集的计算。纯函数，不依赖 torch / pymilvus。"""

import hashlib
import re

_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def content_hash(chunk):
    """chunk 的内容指纹（只覆盖被向量化的文本，标题已包含在 text 的层级前缀里）。"""
    text = (chunk.get("text") or "") + "\n" + (chunk.get("title") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pages_str(pages):
    """[1,1] -> '1'；[1,3] -> '1-3'。"""
    return str(pages[0]) if pages[1] == pages[0] else f"{pages[0]}-{pages[1]}"


def build_row(chunk, dense_vec, sparse_weights, doc_version):
    """把 chunk + 向量组装成 Milvus 行（主键是 chunk_id）。"""
    return {
        "chunk_id": chunk["chunk_id"],
        "title": chunk["title"],
        "level": chunk["level"],
        "path": " > ".join(chunk["path"]),
        "pages": pages_str(chunk["pages"]),
        "char_count": chunk["char_count"],
        "text": chunk["text"],
        "content_hash": content_hash(chunk),
        "doc_version": doc_version,
        "embedding": dense_vec,
        "sparse": sparse_weights,
    }


def collection_name(prefix, version):
    """生成合法的物理集合名（Milvus 要求字母/下划线开头，只含字母数字下划线）。

    ``prefix`` 必须以字母或下划线开头（例如 ``robot_manual_chunks_v``），
    ``version`` 里的非法字符统一替换成下划线。
    """
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", str(version)).strip("_")
    if not suffix:
        suffix = "0"
    name = f"{prefix}{suffix}"
    if not _COLLECTION_NAME_RE.match(name):
        raise ValueError(f"非法集合名: {name}")
    return name


def plan_updates(chunks, existing_hashes, prune=True):
    """对比源 chunk 与库内已有指纹，算出本次要写/要删/可跳过的部分。

    chunks:           源数据（含 chunk_id / text / title）
    existing_hashes:  {chunk_id: content_hash}（来自 Milvus）
    prune:            是否删除"源里已经没有"的 chunk（默认 True）

    返回 {"to_write": [...], "to_delete": [...], "unchanged": [...]}
    """
    seen = set()
    to_write, unchanged = [], []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        seen.add(cid)
        if existing_hashes.get(cid) != content_hash(chunk):
            to_write.append(chunk)
        else:
            unchanged.append(cid)
    to_delete = [cid for cid in existing_hashes if cid not in seen] if prune else []
    return {"to_write": to_write, "to_delete": to_delete, "unchanged": unchanged}
