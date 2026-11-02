"""指令校验：白名单 + 词法扫描 + 参数元数。

三件事：
- 名字在不在手册里。关键字被当函数调用（别的方言那种 DO(1,1)）也要拦。
- 本地定义的函数要豁免，不然模型自己拆出来的流程函数全被判违规。
- 参数个数对不对，签名从手册「格式」栏里解析，解析不出来就不判。

另外有个 safety_regressions，用来挡"把不合规的行注释掉来通过检查"这条路：
安全动作少了一个就拒绝这次修复。

只依赖标准库，不 import torch / langgraph / openai，测试跑得快。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

# ---------- 语言关键字（不是指令） ----------
_KEYWORDS = {
    "if", "else", "then", "do", "while", "for", "end", "step", "to",
    "or", "and", "not", "func", "call", "return", "break", "next",
    "var", "import", "halt",
}
# 允许紧跟括号的结构关键字（其余关键字被当函数调用即违规，例如其它方言的 DO(1,1)）
_CALLABLE_KEYWORDS = {"if", "while", "for", "else"}

# 安全动作标记：修复轮只允许"保持或增加"，不允许减少（大小写不敏感）
_SAFETY_PATTERNS = (
    "waitportregister",   # Modbus 握手应答等待
    "execjz1",            # 气缸到位校验
    "suspend",            # 异常停机
    "motorstate",         # 伺服状态判断
    "motoron",            # 伺服使能
)

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_FUNC_DEF_RE = re.compile(r"^\s*func\s+([A-Za-z_]\w*)\s*\(", re.I | re.M)
_FULLWIDTH_CALL_RE = re.compile(r"[A-Za-z_]\w*\s*[（]")
_IDENT_START_RE = re.compile(r"[A-Za-z_]")
_IDENT_CHAR_RE = re.compile(r"[A-Za-z0-9_]")

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


# ---------------------------------------------------------------- 词法扫描


def strip_comment(line):
    """去掉行尾注释（``//`` 和 ``#``）。字符串内部的那些不算注释。"""
    out = []
    in_str = False
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if in_str:
            out.append(ch)
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and line[i + 1] == "/":
            break
        if ch == "#":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _read_parens(text, start):
    """从 ``text[start] == '('`` 读到配对的右括号，返回 (括号内文本, 右括号后位置)。"""
    depth = 0
    in_str = False
    i, n = start, len(text)
    while i < n:
        ch = text[i]
        if in_str:
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return text[start + 1:], n  # 括号未闭合：取到行尾，交给调用方容忍


def split_args(inner):
    """按顶层逗号切分参数（忽略括号/方括号/字符串内部的逗号）。"""
    if not inner.strip():
        return []
    args, buf = [], []
    depth = 0
    in_str = False
    for ch in inner:
        if in_str:
            buf.append(ch)
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    args.append("".join(buf).strip())
    return [a for a in args if a]


@dataclass(frozen=True)
class Call:
    line: int
    name: str
    arg_count: int
    args_text: str


def iter_calls(code):
    """扫出代码里所有"标识符 + 左括号"的调用，跳过字符串和注释。

    用正则剥注释有两个毛病：logmsg("调用 Clean()") 里的 Clean 会被当成调用；
    logmsg("http://a") 后面的代码会被 // 整段砍掉。这里逐字符走，两个都能处理。
    """
    for line_no, raw in enumerate(code.splitlines(), 1):
        line = strip_comment(raw)
        i, n = 0, len(line)
        in_str = False
        while i < n:
            ch = line[i]
            if ch == '"':
                in_str = not in_str
                i += 1
                continue
            if in_str:
                i += 1
                continue
            if ch == "_" or _IDENT_START_RE.match(ch):
                j = i + 1
                while j < n and _IDENT_CHAR_RE.match(line[j]):
                    j += 1
                name = line[i:j]
                k = j
                while k < n and line[k] in " \t":
                    k += 1
                if k < n and line[k] == "(":
                    inner, end = _read_parens(line, k)
                    yield Call(line=line_no, name=name,
                               arg_count=len(split_args(inner)), args_text=inner)
                    i = end
                    continue
                i = j
                continue
            i += 1


def extract_defined_functions(code):
    """收集本地定义的函数名（``func NAME(...)``）。"""
    return {m.group(1).lower() for m in _FUNC_DEF_RE.finditer(code or "")}


# ---------------------------------------------------------------- 签名解析


def normalize_title(title):
    """``'ReadPortRegister 函数'`` / ``'BusEnable() 函数'`` → ``ReadPortRegister`` / ``BusEnable``。"""
    if not title:
        return ""
    name = str(title).replace("函数", "").replace("（）", "").replace("()", "")
    return re.sub(r"[^A-Za-z0-9_]", "", name)


@dataclass(frozen=True)
class Signature:
    """一条指令的参数签名（来自手册『格式』围栏）。"""

    name: str
    min_args: int
    max_args: int
    variadic: bool
    raw: str

    def accepts(self, arg_count):
        if self.variadic:
            return True
        return self.min_args <= arg_count <= self.max_args


def _signature_from_chunk(chunk):
    """从一个 chunk 的 ```格式``` 围栏里解析出参数签名；解析不出来返回 None。"""
    name = normalize_title(chunk.get("title", ""))
    text = chunk.get("text", "") or ""
    if not name:
        return None
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(", re.I)
    for block in _FENCE_RE.findall(text):
        for line in block.splitlines():
            m = pattern.search(line)
            if not m:
                continue
            inner, _ = _read_parens(line, m.end() - 1)
            args = split_args(inner)
            if not args:
                continue
            variadic = any("..." in a for a in args)
            optional = [a for a in args if a.startswith("[") and a.endswith("]")]
            return Signature(
                name=name,
                min_args=len(args) - len(optional),
                max_args=len(args),
                variadic=variadic,
                raw=line.strip(),
            )
    return None


# ---------------------------------------------------------------- 校验器


class InstructionValidator:
    """白名单取三处的并集：手册标题、格式栏解析出的签名、参考代码里的调用名。

    再加上代码里本地定义的函数（两遍扫描），模型自己拆出来的流程函数才不会被判违规。
    """

    def __init__(self, chunks=(), reference_code="", extra_allowed=()):
        self.chunks = list(chunks or ())
        self.reference_code = reference_code or ""
        self.signatures = {}
        self.allowed = set(_KEYWORDS) | {n.lower() for n in extra_allowed}
        # 手册里真实存在的指令名（与"语言关键字"区分：halt 既是关键字也是手册函数）
        self._manual_names = set()

        for chunk in self.chunks:
            name = normalize_title(chunk.get("title", ""))
            if name:
                self.allowed.add(name.lower())
                self._manual_names.add(name.lower())
            sig = _signature_from_chunk(chunk)
            if sig is not None:
                self.signatures[sig.name.lower()] = sig
                self._manual_names.add(sig.name.lower())

        # 参考代码：既补充白名单，也记录"生产里实际用了几元参数"
        self.observed_arity = {}
        for call in iter_calls(self.reference_code):
            self.allowed.add(call.name.lower())
            self.observed_arity.setdefault(call.name.lower(), set()).add(call.arg_count)

    # ---------- 构造 ----------

    @classmethod
    def from_files(cls, chunks_json, reference_path=None, extra_allowed=()):
        chunks = []
        try:
            chunks = json.loads(Path(chunks_json).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            chunks = []
        ref = ""
        if reference_path and Path(reference_path).exists():
            ref = Path(reference_path).read_text(encoding="utf-8")
        return cls(chunks=chunks, reference_code=ref, extra_allowed=extra_allowed)

    # ---------- 校验 ----------

    def check(self, code, include_warnings=True):
        """返回违规列表（dict，含 kind / severity / message，兼容旧的 line/name/text 键）。"""
        code = code or ""
        defined = extract_defined_functions(code)
        violations = []
        seen = set()

        def add(line_no, name, text, kind, severity, message):
            key = (line_no, name, kind)
            if key in seen:
                return
            seen.add(key)
            violations.append({
                "line": line_no,
                "name": name,
                "text": text,
                "kind": kind,
                "severity": severity,
                "message": message,
            })

        for line_no, raw in enumerate(code.splitlines(), 1):
            if _FULLWIDTH_CALL_RE.search(strip_comment(raw)):
                add(line_no, raw.strip(), raw.strip(), "fullwidth_paren",
                    SEVERITY_ERROR,
                    "使用了全角括号，该行无法执行；请改用半角 ()")

        for call in iter_calls(code):
            low = call.name.lower()
            raw_line = code.splitlines()[call.line - 1].strip()

            if low in _KEYWORDS and low not in self._manual_names:
                if low in _CALLABLE_KEYWORDS:
                    continue
                add(call.line, call.name, raw_line, "keyword_as_call",
                    SEVERITY_ERROR,
                    f"关键字 {call.name} 被当成函数调用（其它机器人方言写法）")
                continue

            if low in defined:
                continue  # 本地定义的函数，定义与调用都合法

            if low not in self.allowed:
                add(call.line, call.name, raw_line, "unknown_instruction",
                    SEVERITY_ERROR,
                    f"指令 {call.name} 不在手册/项目代码白名单内")
                continue

            message = self._arity_message(low, call.arg_count)
            if message and include_warnings:
                add(call.line, call.name, raw_line, "arity",
                    SEVERITY_WARNING, message)

        return violations

    def _arity_message(self, name, arg_count):
        """参数元数校验；信息不足时返回 None（宁可漏报，不误拦）。"""
        observed = self.observed_arity.get(name)
        if observed is not None:
            if len(observed) > 1:
                return None          # 生产代码里本身就是可变参数
            if arg_count in observed:
                return None
        sig = self.signatures.get(name)
        if sig is None or sig.variadic:
            return None
        if sig.accepts(arg_count):
            return None
        return (f"{name} 的格式为 {sig.raw}，期望 {sig.min_args}"
                f"{'' if sig.min_args == sig.max_args else '~' + str(sig.max_args)}"
                f" 个参数，实际传入 {arg_count} 个")

    def errors(self, code):
        return [v for v in self.check(code) if v["severity"] == SEVERITY_ERROR]

    def warnings(self, code):
        return [v for v in self.check(code) if v["severity"] == SEVERITY_WARNING]


# ---------------------------------------------------------------- 安全护栏


def _safety_call_counts(code):
    """数一下真正会被执行的安全动作调用（注释里、字符串里的不算）。"""
    counts = {}
    for call in iter_calls(code or ""):
        low = call.name.lower()
        if low in _SAFETY_PATTERNS:
            counts[low] = counts.get(low, 0) + 1
    return counts


def safety_regressions(before, after):
    """比修复前后安全动作的调用次数，返回被修掉的项 ``{名字: 少了几次}``。

    不能直接 str.count：``// WaitPortRegister(...)`` 里的名字也会被数进去，
    那这层防护等于没有。得靠扫描器数真正会执行的调用。
    """
    before_counts = _safety_call_counts(before)
    after_counts = _safety_call_counts(after)
    dropped = {}
    for name, n_before in before_counts.items():
        n_after = after_counts.get(name, 0)
        if n_after < n_before:
            dropped[name] = n_before - n_after
    return dropped
