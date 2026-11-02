"""状态 reducer 回归测试：覆盖"同一 superstep 多次检索互相覆盖"的静默 bug。"""

import unittest

from src.agent.reducer import MAX_CONTEXT_CHUNKS, merge_chunks, merge_dict


def entity(cid):
    return {"chunk_id": cid, "title": cid, "text": "x" * 10}


class MergeChunksTest(unittest.TestCase):
    def test_two_retrievals_in_one_step_do_not_overwrite(self):
        """一个 superstep 里跑两个检索，后一个不能把前一个的 chunks 盖掉。"""
        first = {"a": entity("a"), "b": entity("b")}
        second = {"c": entity("c")}
        merged = merge_chunks(first, second)
        self.assertEqual(set(merged), {"a", "b", "c"})

    def test_replay_is_idempotent(self):
        """interrupt 恢复会重放节点：同样的增量合并多次结果必须一致。"""
        base = {"a": entity("a")}
        once = merge_chunks(base, {"b": entity("b")})
        twice = merge_chunks(once, {"b": entity("b")})
        self.assertEqual(once, twice)

    def test_cap_evicts_oldest(self):
        merged = {}
        for i in range(MAX_CONTEXT_CHUNKS + 5):
            merged = merge_chunks(merged, {f"c{i}": entity(f"c{i}")})
        self.assertEqual(len(merged), MAX_CONTEXT_CHUNKS)
        self.assertNotIn("c0", merged)          # 最旧的被淘汰
        self.assertIn(f"c{MAX_CONTEXT_CHUNKS + 4}", merged)  # 最新的保留

    def test_recently_used_moves_to_newest(self):
        merged = {}
        for i in range(MAX_CONTEXT_CHUNKS):
            merged = merge_chunks(merged, {f"c{i}": entity(f"c{i}")})
        merged = merge_chunks(merged, {"c0": entity("c0")})  # 再次命中 c0
        merged = merge_chunks(merged, {"new": entity("new")})
        self.assertIn("c0", merged)             # c0 因为刚被用到，不该被淘汰
        self.assertNotIn("c1", merged)          # 该淘汰的是 c1

    def test_handles_none(self):
        self.assertEqual(merge_chunks(None, {"a": entity("a")}), {"a": entity("a")})
        self.assertEqual(merge_chunks({"a": entity("a")}, None), {"a": entity("a")})


class MergeDictTest(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(merge_dict({"a": 1}, {"b": 2}), {"a": 1, "b": 2})
        self.assertEqual(merge_dict(None, {"b": 2}), {"b": 2})


if __name__ == "__main__":
    unittest.main()
