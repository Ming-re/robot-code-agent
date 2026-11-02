"""检查 chunks.json 的结构是否完整。

查得比较细：编号唯一、level 和编号层级对得上、path 长度一致、text 带完整
路径前缀、正文非空、``` 成对、char_count 与 len(text) 一致、pages 合法，
以及同一父节点下的编号是否连续（缺口只告警，手册本来就会跳号）。

    python -m src.verify_chunks
    python -m src.verify_chunks --pdf    # 顺带比一下 PDF 页数
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CHUNKS_JSON, RAW_PDF

_DIGITS_RE = re.compile(r"^\d+$")


def _base_id(chunk_id):
    """去掉去重后缀 '#2'。"""
    return str(chunk_id).split("#")[0]


def _parts(chunk_id):
    return _base_id(chunk_id).split(".")


def verify(chunks):
    """返回 {"errors": [...], "warnings": [...], "stats": {...}}。"""
    errors, warnings = [], []
    seen = defaultdict(list)
    by_parent = defaultdict(list)
    lengths, levels, pages_max = [], Counter(), 0

    for idx, c in enumerate(chunks):
        cid = str(c.get("chunk_id", ""))
        parts = _parts(cid)
        level = c.get("level")
        path = c.get("path") or []
        text = c.get("text") or ""
        pages = c.get("pages") or [0, 0]

        seen[_base_id(cid)].append(idx)
        lengths.append(c.get("char_count") or len(text))
        levels[level] += 1

        if level != len(parts):
            errors.append(f"{cid}: level={level} 与编号层级 {len(parts)} 不一致")
        if len(path) != level:
            errors.append(f"{cid}: path 长度 {len(path)} 与 level {level} 不一致")
        expected_prefix = " > ".join(path)
        if expected_prefix and not text.startswith(expected_prefix):
            errors.append(f"{cid}: text 未以完整层级路径开头")
        body = text[len(expected_prefix):].strip() if expected_prefix else text.strip()
        if not body:
            errors.append(f"{cid}: 正文为空（只有路径前缀）")
        if text.count("```") % 2:
            errors.append(f"{cid}: ``` 围栏未成对")
        if c.get("char_count") not in (None, len(text)):
            errors.append(
                f"{cid}: char_count={c.get('char_count')} 与实际 {len(text)} 不符")
        if not (isinstance(pages, list) and len(pages) == 2
                and 1 <= pages[0] <= pages[1]):
            errors.append(f"{cid}: pages 非法 {pages}")
        pages_max = max(pages_max, pages[1] if len(pages) == 2 else 0)

        if len(parts) >= 2 and _DIGITS_RE.match(parts[-1]):
            by_parent[tuple(parts[:-1])].append(int(parts[-1]))
        elif len(parts) == 1 and _DIGITS_RE.match(parts[-1]):
            by_parent[()].append(int(parts[-1]))

    for cid, idxs in seen.items():
        if len(idxs) > 1:
            errors.append(f"{cid}: chunk_id 重复 {len(idxs)} 次")

    gaps = []
    for parent, nums in sorted(by_parent.items()):
        ordered = sorted(nums)
        if len(set(ordered)) != len(ordered):
            errors.append(f"父节点 {parent}: 同级编号有重复 {ordered}")
            continue
        if ordered[0] != 1:
            gaps.append((parent, f"缺少 1..{ordered[0] - 1}"))
        missing = [n for n in range(1, ordered[-1] + 1) if n not in set(ordered)]
        if missing:
            gaps.append((parent, f"缺少 {missing}"))
    for parent, msg in gaps:
        warnings.append(f"父节点 {parent}: {msg}")

    stats = {
        "total": len(chunks),
        "by_level": dict(sorted(levels.items(), key=lambda kv: (kv[0] or 0))),
        "char_min": min(lengths) if lengths else 0,
        "char_max": max(lengths) if lengths else 0,
        "char_avg": (sum(lengths) // len(lengths)) if lengths else 0,
        "pages_max": pages_max,
        "gap_count": len(gaps),
    }
    return {"errors": errors, "warnings": warnings, "stats": stats}


def pdf_page_count(pdf_path):
    try:
        import pymupdf
    except ImportError:
        return None
    doc = pymupdf.open(str(pdf_path))
    return doc.page_count


def main():
    parser = argparse.ArgumentParser(description="校验 chunks.json 的结构完整性")
    parser.add_argument("--chunks", default=str(CHUNKS_JSON))
    parser.add_argument("--json", default=None, help="把报告写到该路径")
    parser.add_argument("--pdf", action="store_true",
                        help="额外把最大页码与 PDF 总页数比对（需 pymupdf）")
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    report = verify(chunks)

    if args.pdf and Path(RAW_PDF).exists():
        page_count = pdf_page_count(RAW_PDF)
        if page_count:
            covered = report["stats"]["pages_max"]
            ok = covered <= page_count
            print(f"PDF 页数 {page_count}，chunk 覆盖到第 {covered} 页 "
                  f"{'OK' if ok else '超出 PDF 页数!'}")
            if not ok:
                report["errors"].append(
                    f"chunk 页码范围 {covered} 超出 PDF 总页数 {page_count}")

    s = report["stats"]
    print(f"chunk 总数: {s['total']}")
    print(f"层级分布: {s['by_level']}")
    print(f"长度: min={s['char_min']} max={s['char_max']} avg={s['char_avg']}")
    print(f"编号缺口: {s['gap_count']} 处（手册本身可能跳号，仅告警）")
    for w in report["warnings"][:20]:
        print(f"  [warn] {w}")
    for e in report["errors"][:40]:
        print(f"  [error] {e}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入 {args.json}")

    if report["errors"]:
        print(f"\nFAILED: {len(report['errors'])} 个错误")
        raise SystemExit(1)
    print("\nPASSED: 结构校验全部通过")


if __name__ == "__main__":
    main()
