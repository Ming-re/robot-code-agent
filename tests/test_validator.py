"""指令校验器回归测试。

基本都是踩过的坑，改校验逻辑时先跑一遍。
"""

import unittest

from src.agent.validator import (
    InstructionValidator,
    extract_defined_functions,
    iter_calls,
    safety_regressions,
    split_args,
    strip_comment,
)
from src.config import CHUNKS_JSON, PRODUCTION_CODE_PATH


def chunk(cid, title, fmt=None):
    """构造一个带 ```格式``` 围栏的 chunk。"""
    text = f"2 指令详解 > {cid} {title}\n\n定义\n示例\n格式\n"
    if fmt:
        text += f"```\n{fmt}\n```\n"
    return {"chunk_id": cid, "title": title, "level": 3,
            "path": ["2 指令详解", f"{cid} {title}"], "pages": [1, 1],
            "char_count": len(text), "text": text}


CHUNKS = [
    chunk("1.1", "MotorOn 函数", "MotorOn()"),
    chunk("1.2", "MotorOff 函数", "MotorOff()"),
    chunk("1.3", "WritePortRegister 函数", "WritePortRegister(address,value)"),
    chunk("1.4", "ReadPortRegister 函数", "A=ReadPortRegister(address)"),
    chunk("1.5", "LogMsg 函数", "LogMsg(text)"),
    chunk("1.6", "WaitPortRegister 函数",
          "WaitPortRegister(address,value,timeout)"),
]


class TokenizerTest(unittest.TestCase):
    def test_strip_comment_respects_string(self):
        self.assertEqual(
            strip_comment('logmsg("a // b")  x=1'), 'logmsg("a // b")  x=1')
        self.assertEqual(strip_comment("x=1  // 注释"), "x=1  ")
        self.assertEqual(strip_comment("x=1  # 注释"), "x=1  ")

    def test_iter_calls_skips_strings(self):
        names = [c.name for c in iter_calls('logmsg("调用 Clean() 清理")')]
        self.assertEqual(names, ["logmsg"])

    def test_iter_calls_sees_code_after_url_in_string(self):
        # 字符串里的 // 不能当注释，否则后半行的 DoIO 会被漏掉
        names = [c.name for c in iter_calls('logmsg("http://a") DoIO(1,1)')]
        self.assertEqual(names, ["logmsg", "DoIO"])

    def test_iter_calls_ignores_comments(self):
        self.assertEqual(list(iter_calls("// 这里本应 call DoIO(1,1)")), [])

    def test_split_args_nested(self):
        self.assertEqual(split_args('"&Pos",[0,0,偏移,0],Mptp'),
                         ['"&Pos"', "[0,0,偏移,0]", "Mptp"])
        self.assertEqual(split_args(""), [])

    def test_defined_functions(self):
        self.assertEqual(extract_defined_functions("func BuLiao()\nend func"),
                         {"buliao"})


class ValidatorTest(unittest.TestCase):
    def setUp(self):
        self.v = InstructionValidator(chunks=CHUNKS)

    def errors(self, code):
        return [v for v in self.v.check(code) if v["severity"] == "error"]

    def kinds(self, code):
        return sorted({v["kind"] for v in self.v.check(code)})

    # ---- 原先的误拦 ----

    def test_local_function_definition_is_allowed(self):
        """系统提示要求"每个流程一个 func"，模型自己起名字不能算违规。"""
        code = ("func BuLiao()\n"
                "    MotorOn()\n"
                "end func\n"
                "func main()\n"
                "    call BuLiao()\n"
                "end func\n")
        self.assertEqual(self.errors(code), [])

    def test_call_name_inside_string_is_not_a_violation(self):
        code = 'logmsg("请调用 Clean() 清理料盘")\n'
        self.assertEqual(self.errors(code), [])

    # ---- 原先的漏检 ----

    def test_keyword_as_call_is_error(self):
        self.assertIn("keyword_as_call", self.kinds("DO(1,1)\n"))

    def test_unknown_instruction_is_error(self):
        self.assertIn("unknown_instruction", self.kinds("SetDO(2,1)\n"))

    def test_fullwidth_paren_is_error(self):
        self.assertIn("fullwidth_paren", self.kinds("MotorOn（1）\n"))

    def test_arity_mismatch_is_warning(self):
        report = self.v.check("WritePortRegister(1)\n")
        arity = [v for v in report if v["kind"] == "arity"]
        self.assertEqual(len(arity), 1)
        self.assertEqual(arity[0]["severity"], "warning")
        # 只是 warning，不应阻断交付
        self.assertEqual(self.errors("WritePortRegister(1)\n"), [])

    def test_arity_ok(self):
        self.assertEqual(self.v.check("WritePortRegister(1,2)\n"), [])

    def test_arity_skipped_when_reference_is_variadic(self):
        v = InstructionValidator(
            chunks=[chunk("9.9", "LogMsg 函数", "LogMsg(text)")],
            reference_code="logmsg(1)\nlogmsg(1,2)\n")
        self.assertEqual(v.check('logmsg(1,2,3)\n'), [])

    def test_allowed_comes_from_manual_title(self):
        self.assertIn("writeportregister", self.v.allowed)
        self.assertIn("motoron", self.v.allowed)

    def test_signature_parsed_from_fence(self):
        sig = self.v.signatures["writeportregister"]
        self.assertEqual((sig.min_args, sig.max_args), (2, 2))

    def test_comment_mention_is_not_a_violation(self):
        self.assertEqual(self.errors("// 应该 call DoIO(1,1) 但手册没有\n"), [])


class SafetyGuardTest(unittest.TestCase):
    def test_commented_out_safety_action_is_detected(self):
        before = ("WaitPortRegister(12,0,-1)\n"
                  "call execjz1(CY1,CX1,CX2)\n"
                  "Suspend()\n")
        after = ("// WaitPortRegister(12,0,-1)\n"
                 "call execjz1(CY1,CX1,CX2)\n"
                 "Suspend()\n")
        dropped = safety_regressions(before, after)
        self.assertEqual(dropped, {"waitportregister": 1})

    def test_equivalent_rewrite_is_allowed(self):
        before = "MOTORON()\ncall execjz1(a,b,c)\n"
        after = "motoron()\ncall execjz1(a,b,c)\n"
        self.assertEqual(safety_regressions(before, after), {})


class ManualVersusKeywordTest(unittest.TestCase):
    """关键字分支不能盖过手册里真实存在的同名函数（如 Halt 函数）。"""

    def test_manual_function_named_like_keyword_is_allowed(self):
        v = InstructionValidator(chunks=[chunk("9.1", "Halt 函数")])
        self.assertEqual(v.errors("halt()\n"), [])

    def test_pure_dialect_keyword_still_flagged(self):
        v = InstructionValidator(chunks=[chunk("9.1", "Halt 函数")])
        self.assertIn("keyword_as_call",
                      [x["kind"] for x in v.check("DO(1,1)")])


@unittest.skipUnless(PRODUCTION_CODE_PATH.exists() and CHUNKS_JSON.exists(),
                     "缺少 main.prg 或 chunks.json")
class ProductionCodeTest(unittest.TestCase):
    """生产代码本身要能过校验器，过不了说明白名单漏了合法用法。"""

    def test_production_code_has_no_errors(self):
        v = InstructionValidator.from_files(CHUNKS_JSON, PRODUCTION_CODE_PATH)
        code = PRODUCTION_CODE_PATH.read_text(encoding="utf-8")
        errors = v.errors(code)
        self.assertEqual([(e["line"], e["name"]) for e in errors], [])

    def test_allowlist_size(self):
        v = InstructionValidator.from_files(CHUNKS_JSON, PRODUCTION_CODE_PATH)
        # 手册标题 + 项目代码调用名 + 关键字，去重后约 200+ 条
        self.assertGreater(len(v.allowed), 180)
        self.assertGreater(len(v.signatures), 100)


if __name__ == "__main__":
    unittest.main()
