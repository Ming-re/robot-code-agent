"""入库计划回归测试：幂等（主键 + 指纹 + 差集）。"""

import unittest

from src.ingest_plan import (
    build_row,
    collection_name,
    content_hash,
    pages_str,
    plan_updates,
)


def chunk(cid, body="正文"):
    return {"chunk_id": cid, "title": f"{cid} 函数", "level": 3,
            "path": ["2 指令详解", f"{cid} 函数"], "pages": [3, 4],
            "char_count": 10, "text": f"2 指令详解 > {cid} 函数\n\n{body}"}


class ContentHashTest(unittest.TestCase):
    def test_stable(self):
        self.assertEqual(content_hash(chunk("2.1")), content_hash(chunk("2.1")))

    def test_changes_with_text(self):
        self.assertNotEqual(content_hash(chunk("2.1")),
                            content_hash(chunk("2.1", "改了")))

    def test_changes_with_title(self):
        a = chunk("2.1")
        b = dict(a, title="别的标题")
        self.assertNotEqual(content_hash(a), content_hash(b))


class PlanUpdatesTest(unittest.TestCase):
    def test_first_run_writes_everything(self):
        plan = plan_updates([chunk("2.1"), chunk("2.2")], {})
        self.assertEqual([c["chunk_id"] for c in plan["to_write"]],
                         ["2.1", "2.2"])
        self.assertEqual(plan["to_delete"], [])

    def test_second_run_writes_nothing(self):
        """重复入库必须是无操作（幂等的核心断言）。"""
        source = [chunk("2.1"), chunk("2.2")]
        existing = {c["chunk_id"]: content_hash(c) for c in source}
        plan = plan_updates(source, existing)
        self.assertEqual(plan["to_write"], [])
        self.assertEqual(plan["to_delete"], [])
        self.assertEqual(sorted(plan["unchanged"]), ["2.1", "2.2"])

    def test_only_changed_is_rewritten(self):
        source = [chunk("2.1"), chunk("2.2", "新内容")]
        existing = {"2.1": content_hash(chunk("2.1")),
                    "2.2": content_hash(chunk("2.2"))}
        plan = plan_updates(source, existing)
        self.assertEqual([c["chunk_id"] for c in plan["to_write"]], ["2.2"])
        self.assertEqual(plan["unchanged"], ["2.1"])

    def test_removed_chunk_is_deleted(self):
        source = [chunk("2.1")]
        existing = {"2.1": content_hash(chunk("2.1")),
                    "2.9": content_hash(chunk("2.9"))}
        plan = plan_updates(source, existing, prune=True)
        self.assertEqual(plan["to_delete"], ["2.9"])
        plan_keep = plan_updates(source, existing, prune=False)
        self.assertEqual(plan_keep["to_delete"], [])


class NamingTest(unittest.TestCase):
    def test_collection_name(self):
        self.assertEqual(collection_name("robot_manual_chunks_v", "20241101_1200"),
                         "robot_manual_chunks_v20241101_1200")

    def test_illegal_chars_are_sanitized(self):
        name = collection_name("p_v", "2024-11-01 12:00")
        self.assertRegex(name, r"^[A-Za-z_][A-Za-z0-9_]*$")

    def test_empty_version(self):
        self.assertEqual(collection_name("p_v", ""), "p_v0")


class BuildRowTest(unittest.TestCase):
    def test_row_shape(self):
        row = build_row(chunk("2.1"), [0.1, 0.2], {1: 0.5}, "20241101")
        self.assertEqual(row["chunk_id"], "2.1")
        self.assertEqual(row["pages"], "3-4")
        self.assertEqual(row["doc_version"], "20241101")
        self.assertEqual(row["content_hash"], content_hash(chunk("2.1")))
        self.assertEqual(row["path"], "2 指令详解 > 2.1 函数")

    def test_pages_str(self):
        self.assertEqual(pages_str([5, 5]), "5")
        self.assertEqual(pages_str([5, 7]), "5-7")


if __name__ == "__main__":
    unittest.main()
