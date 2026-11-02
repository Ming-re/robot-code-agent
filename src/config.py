"""全局配置: 数据路径、向量模型、Milvus 连接与集合参数。"""

import os
from pathlib import Path

# ---- 路径 ----
ROOT_DIR = Path(__file__).resolve().parents[1]
# 数据默认放 ./data；想放到项目目录外面就设环境变量 ROBOT_DATA_DIR
DATA_DIR = Path(os.environ.get("ROBOT_DATA_DIR") or (ROOT_DIR / "data"))
RAW_PDF = DATA_DIR / "raw" / "manual.pdf"
CHUNKS_JSON = DATA_DIR / "chunks" / "chunks.json"
CHUNKS_MD = DATA_DIR / "chunks" / "chunks.md"

# ---- 向量模型 (bge-m3) ----
EMBEDDING_MODEL_DIR = ROOT_DIR / "models" / "embedding" / "bge-m3"
EMBEDDING_DIM = 1024          # bge-m3 dense 向量维度
EMBED_BATCH_SIZE = 32         # 编码 batch 大小

# ---- Milvus ----
MILVUS_URI = "http://localhost:19530"
# COLLECTION_NAME 是别名(alias)，应用只查这个名字,
# 物理集合按版本命名(robot_manual_chunks_v20240101_1200), 切换 alias 即蓝绿发布。
COLLECTION_NAME = "robot_manual_chunks"
COLLECTION_PREFIX = "robot_manual_chunks_v"

# ---- 混合检索 + 重排序（单查询管线）----
HYBRID_TOP_K = 10     # 混合检索(dense+sparse RRF)候选数，供 reranker 重排
FINAL_TOP_K = 1       # 重排序后最终返回条数
# 重排序用的是"融合后的候选池", 池子必须远大于最终返回数, 否则召回被截断
RERANK_POOL_K = 50

# ---- Agent 上下文 ----
MAX_CONTEXT_CHUNKS = 24       # 会话内累积 chunk 上限（超出按 LRU 淘汰）
SESSION_TTL_SECONDS = 2 * 3600  # 服务端会话元信息保留时长

# ---- LLM 代码生成 (DeepSeek) ----
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.1   # 代码生成用低温, 保证确定性
DEEPSEEK_MAX_TOKENS = 8192
GENERATED_CODE_DIR = DATA_DIR / "generated"
# 项目现有生产代码, 生成时作为风格/寄存器约定参考注入提示词
PRODUCTION_CODE_PATH = ROOT_DIR / "main.prg"

# ---- 评估 ----
EVAL_DIR = ROOT_DIR / "eval"
EVAL_QUERIES = EVAL_DIR / "queries.jsonl"
EVAL_BASELINE = EVAL_DIR / "baseline.json"
EVAL_REPORT_DIR = EVAL_DIR / "reports"
# RRF 分数低于该值视为"没有可信命中"（用于负例/拒答统计, 见 eval/README）
RRF_ANSWER_FLOOR = 0.02

# ---- Agent 服务 ----
API_HOST = "127.0.0.1"
API_PORT = 8000
