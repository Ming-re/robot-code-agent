"""切块数据完整性测试：把"187 chunk、编号零缺失、正文零丢失"变成断言。"""

import json
import unittest
from pathlib import Path

from src.config import CHUNKS_JSON, ROOT_DIR
from src.verify_chunks import verify


def make_chunk(cid, level, path=None, body="正文", **over):
    path = path or [f"{cid} 标题"]
    prefix = " > ".join(path)
    text = f"{prefix}\n\n{body}"
    item = {"chunk_id": cid, "heading": f"{cid} 标题", "title": "标题",
            "level": level, "path": path, "pages": [1, 1],
            "char_count": len(text), "text": text}
    item.update(over)
    return item


class VerifierTest(unittest.TestCase):
    def setUp(self):
        self.good = [
            make_chunk("1", 1, ["1 指令详解"]),
            make_chunk("1.1", 2, ["1 指令详解", "1.1 数学"]),
            make_chunk("1.1.1", 3, ["1 指令详解", "1.1 数学", "1.1.1 Abs"]),
            make_chunk("1.1.2", 3, ["1 指令详解", "1.1 数学", "1.1.2 Acos"]),
        ]

    def test_clean_data_passes(self):
        report = verify(self.good)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["stats"]["total"], 4)

    def test_level_mismatch(self):
        bad = self.good + [make_chunk("1.1.3.1", 3)]
        report = verify(bad)
        self.assertTrue(any("level" in e for e in report["errors"]))

    def test_prefix_missing(self):
        bad = [make_chunk("1.1", 1, ["别的前缀"])]
        bad[0]["path"] = ["1.1 标题"]
        bad[0]["text"] = "错误的开头\n\n正文"
        bad[0]["char_count"] = len(bad[0]["text"])
        report = verify(bad)
        self.assertTrue(any("层级路径" in e for e in report["errors"]))

    def test_empty_body(self):
        item = make_chunk("1", 1, ["1 标题"], body="   ")
        item["text"] = "1 标题\n\n   "
        item["char_count"] = len(item["text"])
        report = verify([item])
        self.assertTrue(any("正文为空" in e for e in report["errors"]))

    def test_unbalanced_fence(self):
        report = verify([make_chunk("1", 1, ["1 标题"], body="```\nA=1\n")])
        self.assertTrue(any("围栏" in e for e in report["errors"]))

    def test_duplicate_id(self):
        report = verify([make_chunk("1", 1, ["1 标题"]),
                         make_chunk("1", 1, ["1 标题"])])
        self.assertTrue(any("重复" in e for e in report["errors"]))

    def test_gap_is_warning_not_error(self):
        report = verify([
            make_chunk("1", 1, ["1 标题"]),
            make_chunk("1.1", 2, ["1 标题", "1.1 a"]),
            make_chunk("1.3", 2, ["1 标题", "1.3 c"]),
        ])
        self.assertEqual(report["errors"], [])
        self.assertTrue(any("缺少" in w for w in report["warnings"]))

    def test_char_count_mismatch(self):
        item = make_chunk("1", 1, ["1 标题"])
        item["char_count"] = 1
        report = verify([item])
        self.assertTrue(any("char_count" in e for e in report["errors"]))

    def test_illegal_pages(self):
        report = verify([make_chunk("1", 1, ["1 标题"], pages=[3, 2])])
        self.assertTrue(any("pages" in e for e in report["errors"]))


@unittest.skipUnless(Path(CHUNKS_JSON).exists(), "缺少 chunks.json")
class RealDataTest(unittest.TestCase):
    """对真实数据跑一遍：这是文档里"编号零缺失、正文零丢失"的可执行版本。"""

    @classmethod
    def setUpClass(cls):
        cls.chunks = json.loads(Path(CHUNKS_JSON).read_text(encoding="utf-8"))
        cls.report = verify(cls.chunks)

    def test_no_errors(self):
        self.assertEqual(self.report["errors"], [])

    def test_no_numbering_gaps(self):
        self.assertEqual(self.report["stats"]["gap_count"], 0)

    def test_every_chunk_has_body(self):
        for chunk in self.chunks:
            prefix = " > ".join(chunk["path"])
            self.assertTrue(chunk["text"].startswith(prefix))
            self.assertTrue(chunk["text"][len(prefix):].strip())


if __name__ == "__main__":
    unittest.main()
