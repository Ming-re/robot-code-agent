"""调 DeepSeek 生成和修正机器人程序。

prompt 里把不变的参考代码放前面、需求放后面。DeepSeek 的前缀缓存按前缀
完全一致匹配，顺序摆反了就一次都命中不了。
"""

import os
import re

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from ..config import (
    CHUNKS_JSON,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    PRODUCTION_CODE_PATH,
)
from .validator import InstructionValidator, safety_regressions

# chunk 文本中 ``` 围栏块（格式签名/示例代码）
_FENCE_RE = re.compile(r"```\n(.*?)```", re.S)

FIX_SYSTEM_PROMPT = """你是工业机器人代码修正助手。用户代码中部分指令名不在《机器人指令手册》和项目参考代码中。

修正规则（严格遵守）：
1. 能从「可用指令清单」或项目参考代码中找到**语义等价**指令的，直接替换修正；
2. **禁止用注释掉整行来"解决"不合规**——这会删掉真实功能，在物理设备上等于制造危险；
3. 找不到等价指令时，保留原行不动，在该行上方加一行 `// UNRESOLVED: 原因`；
4. 任何安全动作（WaitPortRegister 握手、execjz1 气缸到位校验、Suspend 报警停机、
   MotorState/MotorOn 伺服使能）**绝对不允许删除或注释掉**；
5. 只输出修正后的完整代码，不要解释文字，不要 markdown 围栏。"""

SYSTEM_PROMPT = """你是资深工业机器人编程专家，精通《机器人指令手册》所有指令，并熟悉项目现有代码风格。
任务：根据自然语言需求与参考指令文档，生成完整可运行的机器人程序。

## 项目代码规范（严格遵循，参照「项目参考代码」）
1. 结构: 全局变量 VAR 区(// 或 # 注释说明用途) → IO 变量定义区(输入 CX 前缀/输出 CY 前缀,
   用 VAR 封装 IO 号) → 点位定义(VAR MPHome=point("PHome")) → 按流程拆分功能函数
   (初始化 Init/取料 Pick/放料 Push/抛料 NG/拍照 Photo/标定 calib 等, 每个流程一个 func)
   → func main() 状态机循环(if/else if 按优先级分支, call 调用子函数)
2. 指令名不区分大小写, 项目习惯小写: ptp/ptprel/line/on/off/out/in/waittime/logmsg
3. 指令以「可用指令清单」为准; 项目参考代码中出现过的写法(如 point/ptprel/ReadNumber/
   WriteNumber/DynOver/exec 等)视为本项目有效用法; 禁止 LIN/MOVL/DO/DI/SetDO 等
   手册与参考代码之外的指令
4. Modbus 交互: ReadPortRegister/WritePortRegister/WaitPortRegister 直接读寄存器号,
   读信号集中在 func modbus() 里赋值给 M 前缀变量; 输出封装 func Modbus_out(address,value)
   并 call 调用; 握手模式: 置位→WaitPortRegister 等待应答→复位
5. 气缸动作必须带到位校验: 检测动点/原点输入, 超时(TimeStamp 计时)logmsg 报警并 Suspend(),
   参照 execjz1 模式
6. 点位: 现场示教名 point("PHome")/point("取料点") 等; 接近/抬起用 ptprel("&Pos",[0,0,偏移,0],Mptp)
   相对偏移; 运动带速度参数(Mptp/MLine 变量)
7. 日志: 关键流程 logmsg(time(),"进入取料") 记录, 节拍用 TimeStamp() 计算;
   后台持续上传信号用 exec(线程函数)(while true do ... waittime 循环)
8. 变量命名 M 前缀 + 含义(如 MPick1/Mxizui_OK/MPhoto); 自定义函数名可按语义自行取,
   但必须先 func 定义、再调用; 缺失细节用合理默认值并标 // TODO: 现场确认
9. 只输出程序代码, 不要解释文字和 markdown 围栏
"""


def build_catalog(chunks):
    """从 chunk 文本提取指令签名清单: "chunk_id 标题: 签名行"。"""
    items = []
    for c in chunks:
        m = _FENCE_RE.search(c["text"])
        if not m:
            continue
        sig = m.group(1).strip().split("\n")[0]
        items.append(f"{c['chunk_id']} {c['title']}: {sig}")
    return items


class CodeGenerator:
    def __init__(self, api_key=None, base_url=DEEPSEEK_BASE_URL,
                 model=DEEPSEEK_MODEL, validator=None):
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )
        self._model = model
        # 项目现有生产代码, 注入提示词作为风格与寄存器约定参考
        self._ref_code = ""
        if PRODUCTION_CODE_PATH.exists():
            self._ref_code = PRODUCTION_CODE_PATH.read_text(encoding="utf-8")
        # 指令校验: 手册『格式』签名 + 标题 + 项目代码调用名 + 关键字
        self._validator = validator or InstructionValidator.from_files(
            CHUNKS_JSON, PRODUCTION_CODE_PATH)

    # ---------- 校验（委托 validator） ----------

    @property
    def validator(self):
        return self._validator

    @property
    def allowed(self):
        return self._validator.allowed

    def check_report(self, code):
        """完整报告：含 error 与 warning（warning 不会阻断交付）。"""
        return self._validator.check(code)

    def check_instructions(self, code):
        """兼容旧接口：只返回会导致交付失败的 error 级违规。"""
        return self._validator.errors(code)

    @staticmethod
    def safety_regressions(before, after):
        return safety_regressions(before, after)

    # ---------- 修复 ----------

    async def verify_and_fix(self, code):
        """发现不合规指令时调用 LLM 修正, 返回 (修正后代码, 是否修改)。

        注意：是否接受这次修复由调用方决定（见 ``safety_regressions`` 与复检）。
        """
        violations = self.check_instructions(code)
        if not violations:
            return code, False
        desc = "\n".join(f"- 第{v['line']}行: {v['text']} (违规指令: {v['name']})"
                         for v in violations)
        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=FIX_SYSTEM_PROMPT),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"## 不合规指令\n{desc}\n\n## 待修正代码\n{code}"),
        ]
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            temperature=0,
            max_tokens=DEEPSEEK_MAX_TOKENS,
        )
        fixed = resp.choices[0].message.content or ""
        return fixed, fixed != code

    @staticmethod
    def unresolved_markers(code):
        """修复轮里模型为自己标记"无法解决"的行数。"""
        return len(re.findall(r"UNRESOLVED", code or "", re.I))

    # ---------- 生成 ----------

    def _static_prefix(self):
        """不随请求变化的参考代码块（放在 prompt 最前面，便于命中前缀缓存）。"""
        if not self._ref_code:
            return ""
        return ("## 项目参考代码（同机型现有生产程序，结构、指令用法、命名习惯均参照下面的写法）\n"
                f"{self._ref_code}")

    def _build_messages(self, requirement, chunks):
        """构建生成用的 messages：静态块在前、变量块在后（前缀缓存友好）。"""
        chunks = list({c["chunk_id"]: c for c in chunks}.values())
        catalog = "\n".join(f"- {item}" for item in build_catalog(chunks))
        ref = "\n\n".join(
            f"### [{c['chunk_id']}] {c['title']}\n{c['text']}" for c in chunks
        )
        messages = [ChatCompletionSystemMessageParam(
            role="system", content=SYSTEM_PROMPT)]
        static_prefix = self._static_prefix()
        if static_prefix:
            messages.append(ChatCompletionUserMessageParam(
                role="user", content=static_prefix))
        messages.append(ChatCompletionUserMessageParam(
            role="user",
            content="\n\n".join([
                f"## 用户需求\n{requirement}",
                f"## 可用指令清单\n{catalog}",
                f"## 参考指令文档\n{ref}",
            ]),
        ))
        return messages

    async def stream_generate(self, requirement, chunks):
        """流式生成: 逐段产出 ("delta", 文本) 事件, 最后产出 ("done", tokens)。"""
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(requirement, chunks),
            stream=True,
            stream_options={"include_usage": True},
            temperature=DEEPSEEK_TEMPERATURE,
            max_tokens=DEEPSEEK_MAX_TOKENS,
        )
        usage = None
        async for chunk in resp:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield "delta", delta.content
            if chunk.usage:
                usage = chunk.usage
        tokens = ((usage.prompt_tokens, usage.completion_tokens)
                  if usage else (0, 0))
        yield "done", tokens
