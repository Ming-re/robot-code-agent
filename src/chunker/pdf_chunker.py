"""
按标题切分《机器人指令手册》PDF。
切分规则:
- 每个标题 + 其正文（到下一个标题之前的所有内容）为一个 chunk
- 没有直接正文的标题（后面紧跟子标题）不单独成 chunk，但仍保留在子 chunk 的层级路径中
- chunk 的 text 以完整层级路径开头（便于后续向量化检索），metadata 中保留标题、层级、页码等
- 「格式/程序示例」等代码字段用 ``` 围栏包裹，与描述文字区分开并保留换行

输出: chunks.json（结构化数据）+ chunks.md（Markdown 版本），每项包含 chunk_id / title / level / path / pages / text / char_count，
可直接用于后续写入 Milvus 向量库。
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
import pymupdf

# 项目根目录（src/chunker/ 上两级）
_ROOT = Path(__file__).resolve().parents[2]
# 和 src/config.py 保持一致：数据默认放 ./data，可用 ROBOT_DATA_DIR 指到别处
_DATA_DIR = Path(os.environ.get("ROBOT_DATA_DIR") or (_ROOT / "data"))
DEFAULT_PDF = _DATA_DIR / "raw" / "manual.pdf"
DEFAULT_JSON = _DATA_DIR / "chunks" / "chunks.json"
DEFAULT_MD = _DATA_DIR / "chunks" / "chunks.md"

# 行重建: 同一视觉行内各 span 中心 y 的容差（正文行距约 24pt，字号 12pt）
LINE_Y_TOL = 4.5
# 编号标题行: "2.2.5 打开伺服函数" / "2 指令详解"
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s*(.*)$")
BOLD_FLAGS = 16  # PyMuPDF flags 第 16 位表示粗体（此处以字体名为准，双保险）

# 正文字段标签: "格式/程序示例" 为代码字段（有编号变体如 "格式 1"/"程序示例2"），
# "//格式 1" 是代码内注释（以 // 开头，不会被匹配）；"示例程序" 是 "程序示例" 的同义标签
CODE_LABEL_RE = re.compile(r"^(格式|程序示例)\s*\d*$")
PROSE_LABELS = {"定义", "参数", "返回值", "说明", "结果", "注意",
                "注意事项", "注意事项："}

def _is_code_label(text):
    """是否为代码字段标签（格式/程序示例，可带编号）。"""
    return text == "示例程序" or bool(CODE_LABEL_RE.match(text))

def _is_prose_label(text):
    return text in PROSE_LABELS

def extract_lines(page):
    """从一页中重建文本行。

    PDF 中同一行常被拆成多个 block/span（水印与排版导致），
    这里把所有 span 按中心 y 聚类成行，行内按 x 排序。
    返回: [{'text': str, 'spans': [span, ...]}, ...]
    """
    d = page.get_text("dict", sort=True)
    groups = {}
    for b in d["blocks"]:
        if b["type"] != 0:  # 0 = 文本块，跳过图片（水印是图片，不影响）
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                cy = (span["bbox"][1] + span["bbox"][3]) / 2
                key = None
                for k in groups:
                    if abs(k - cy) < LINE_Y_TOL:
                        key = k
                        break
                if key is None:
                    key = cy
                    groups[key] = []
                groups[key].append(span)

    lines = []
    for cy in sorted(groups):
        spans = sorted(groups[cy], key=lambda s: s["bbox"][0])
        # 拼接 span 文本; 相邻 span 之间若无空白且有间距, 补一个空格
        parts = []
        for i, s in enumerate(spans):
            if i > 0:
                prev = spans[i - 1]
                gap = s["bbox"][0] - prev["bbox"][2]
                if gap > 0.5 and not prev["text"].endswith((" ", "\t")) \
                        and not s["text"].startswith((" ", "\t")):
                    parts.append(" ")
            parts.append(s["text"])
        text = "".join(parts).strip()
        if text:
            lines.append({"text": text, "spans": spans})
    return lines

def classify_heading(line):
    """判断一行是否为编号标题。是则返回 (编号, 标题文字, 层级)，否则返回 None。"""
    m = HEADING_RE.match(line["text"])
    if not m:
        return None
    num, rest = m.group(1), m.group(2).strip()
    if not num:
        return None
    # 编号后紧跟标点（如 "1 . xxx"、"5 ：xxx"）是列表项/示例输出，不是标题
    if rest and rest[0] in ".,、。:：;；)）]】":
        return None
    # 编号部分对应的 span 必须为粗体（这是区分标题与正文数字行的核心依据）
    acc, i = "", 0
    spans = line["spans"]
    while i < len(spans) and num not in acc:
        acc += spans[i]["text"]
        i += 1
    num_spans = spans[:i]
    if not num_spans or not all(
        "Bold" in s["font"] or (s["flags"] & BOLD_FLAGS) for s in num_spans
    ):
        return None
    return num, rest, num.count(".") + 1

def render_lines(lines):
    """渲染 chunk 正文: 代码字段（格式/程序示例）用 ``` 围栏包裹，其余按行保留。

    "格式 N" 编号行有两种含义: 多格式函数的字段标签（如 2.3.1），
    或「说明」里的引用文字。不在代码块中时，用下一行是否以 ASCII 开头来区分。
    """
    out, code_buf, in_code = [], [], False
    numbered_format_re = re.compile(r"^格式\s*\d+$")
    ascii_start_re = re.compile(r"^[A-Za-z(\[]")

    def flush_code():
        if code_buf:
            out.append("```")
            out.extend(code_buf)
            out.append("```")
            code_buf.clear()

    def next_nonempty(i):
        for j in range(i + 1, len(lines)):
            if lines[j].strip():
                return lines[j]
        return ""

    for i, ln in enumerate(lines):
        if _is_prose_label(ln):
            if in_code:
                flush_code()
            out.append(ln)
            in_code = False
        elif _is_code_label(ln):
            # 编号 "格式 N" 不在代码块中且下一行是中文描述 -> 说明里的引用文字
            if numbered_format_re.match(ln) and not in_code:
                nxt = next_nonempty(i)
                if nxt and not ascii_start_re.match(nxt):
                    out.append(ln)
                    continue
            if in_code:
                flush_code()
            out.append(ln)
            in_code = True
        elif in_code:
            code_buf.append(ln)
        else:
            out.append(ln)
    flush_code()
    return "\n".join(out)

@dataclass
class _Draft:
    """切分过程中的临时 chunk 结构（用 dataclass 避免异构 dict 引发的类型联合推断）。"""
    id: str
    title: str
    level: int
    path: list[str]
    page_start: int
    lines: list[str] = field(default_factory=list)
    page_end: int | None = None

def chunk_pdf(pdf_path):
    """按标题切分 PDF，返回 chunk 列表。"""
    doc = pymupdf.open(str(pdf_path))
    chunks = []
    stack: list[tuple[str, str, int]] = []   # 祖先标题栈: [(编号, 标题, 层级)]
    cur: _Draft | None = None                # 当前 chunk
    used_ids: set[str] = set()

    def finish_cur():
        nonlocal cur
        if cur is None:
            return
        if not cur.lines:  # 无直接正文的标题（后面紧跟子标题）不单独成 chunk
            cur = None
            return
        # chunk_id 去重（编号重复时追加序号）
        cid = cur.id
        if cid in used_ids:
            n = 2
            while f"{cid}#{n}" in used_ids:
                n += 1
            cid = f"{cid}#{n}"
        used_ids.add(cid)
        page_end = cur.page_end if cur.page_end is not None else cur.page_start
        path_text = " > ".join(cur.path)
        text = f"{path_text}\n\n" + render_lines(cur.lines)
        chunks.append({
            "chunk_id": cid,
            "heading": f"{cur.id} {cur.title}".strip(),
            "title": cur.title,
            "level": cur.level,
            "path": cur.path,
            "pages": [cur.page_start, page_end],
            "char_count": len(text),
            "text": text,
        })
        cur = None

    for pno, page in enumerate(doc):
        page_no = pno + 1
        for line in extract_lines(page):
            heading = classify_heading(line)
            if heading:
                num, title, level = heading
                finish_cur()
                # 更新祖先栈: 弹出层级 >= 当前层级的标题
                while stack and stack[-1][2] >= level:
                    stack.pop()
                stack.append((num, title, level))
                path = [f"{n} {t}".strip() for n, t, _ in stack]
                cur = _Draft(
                    id=num,
                    title=title,
                    level=level,
                    path=path,
                    page_start=page_no,
                )
            elif cur is not None:
                cur.lines.append(line["text"])
                cur.page_end = page_no

    finish_cur()
    return chunks

def _md_pad_fences(text):
    """在 ``` 围栏前后补空行，保证各种 Markdown 渲染器都能正确识别代码块。"""
    text = re.sub(r"\n```", "\n\n```", text)
    text = re.sub(r"```\n", "```\n\n", text)
    return text

def write_markdown(chunks, md_path):
    """将 chunks 写入 Markdown 文件，标题层级对应 chunk 层级。"""
    lines = [f"# 机器人指令手册（共 {len(chunks)} 个 chunk）", ""]
    for c in chunks:
        lines.append("#" * min(c["level"], 6) + " " + c["heading"])
        lines.append("")
        pages = c["pages"]
        page_str = str(pages[0]) if pages[1] == pages[0] \
            else f"{pages[0]}-{pages[1]}"
        lines.append(f"*页码: {page_str}*")
        lines.append("")
        lines.append(_md_pad_fences(c["text"]))
        lines.append("")
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="按标题切分 PDF 并输出 chunks.json")
    parser.add_argument("pdf", nargs="?", default=str(DEFAULT_PDF),
                        help="PDF 文件路径")
    parser.add_argument("-o", "--output", default=str(DEFAULT_JSON),
                        help="输出 JSON 路径")
    parser.add_argument("-m", "--markdown", default=str(DEFAULT_MD),
                        help="输出 Markdown 路径")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"找不到文件: {pdf_path}")

    chunks = chunk_pdf(pdf_path)

    out = Path(args.output)
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    md_out = Path(args.markdown)
    write_markdown(chunks, md_out)

    # 统计信息
    from collections import Counter
    by_level = Counter(c["level"] for c in chunks)
    lengths = [c["char_count"] for c in chunks]
    print(f"PDF: {pdf_path} -> {out}, {md_out}")
    print(f"chunk 总数: {len(chunks)}")
    print(f"按层级分布: {dict(sorted(by_level.items()))}")
    if lengths:
        print(f"chunk 长度: min={min(lengths)} max={max(lengths)} "
              f"avg={sum(lengths) // len(lengths)}")
    print("\n前 5 个 chunk 预览:")
    for c in chunks[:5]:
        print(f"  [{c['chunk_id']}] L{c['level']} p{c['pages']} "
              f"{c['title']} ({c['char_count']} 字)")

if __name__ == "__main__":
    main()



