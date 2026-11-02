"""状态字段怎么合并。

messages 往后追加；chunks 按 chunk_id 合并，顺便做个 LRU 截断别让它无限长。

这里以前还有个 merge_dict，用来存 completed[tool_call_id]（防 interrupt 重放）。
后来发现那个记忆机制根本生效不了——它写回 state 的那一行在函数末尾，
interrupt 抛异常时函数已经退出了。现在 ask 拆成独立节点，不需要记忆，
merge_dict 也就一起删掉了。
"""

# chunks 上下文上限：超过后淘汰最久未使用的（控制检查点体积与生成 prompt 长度）
MAX_CONTEXT_CHUNKS = 24


def merge_chunks(left, right):
    """按 chunk_id 合并检索结果。

    - 同一 chunk_id 后写覆盖，并把它挪到"最近使用"的位置（dict 插入序即 LRU 序）；
    - 超过 MAX_CONTEXT_CHUNKS 淘汰最旧的，避免会话越长上下文越大；
    - 幂等：同样的 right 合并多次结果不变（节点重放安全）。
    """
    merged = dict(left or {})
    for chunk_id, entity in (right or {}).items():
        merged.pop(chunk_id, None)
        merged[chunk_id] = entity
    while len(merged) > MAX_CONTEXT_CHUNKS:
        merged.pop(next(iter(merged)))
    return merged
