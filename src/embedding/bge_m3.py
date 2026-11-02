"""bge-m3 三路输出编码器: dense + sparse + colbert（基于 FlagEmbedding 加载本地模型）。

- dense:   1024 维归一化向量，用于 Milvus ANN 检索
- sparse:  词权重 {token_id: weight}，用于 Milvus 倒排稀疏检索（混合检索的另一路）
- colbert: 词级向量，用于候选重排序（ColBERT MaxSim，词级交互强于双塔点积）

本地模型目录含 sparse_linear.pt / colbert_linear.pt 两个头，FlagEmbedding 的
BGEM3FlagModel 可直接加载。
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
from FlagEmbedding import BGEM3FlagModel  # noqa: E402


class BgeM3Encoder:
    def __init__(self, model_dir, max_length=2048):
        self._model = BGEM3FlagModel(str(model_dir), use_fp16=False)
        self._max_length = max_length

    def encode_query(self, text, with_colbert=True):
        """查询编码。

        ``with_colbert=False`` 时不算 ColBERT 词级向量，Agent 的检索路径用不到它，
        算了再丢掉等于白烧 CPU（实测单次查询多 10%~30% 延迟）。
        """
        return self._encode([text], batch_size=1, return_colbert=with_colbert)

    def encode_docs(self, texts, batch_size=16):
        """文档编码（入库用）: dense + sparse，不需要 colbert。"""
        return self._encode(list(texts), batch_size=batch_size,
                            return_colbert=False)

    def encode_for_rerank(self, texts, batch_size=16):
        """候选文档编码（重排序用）: 只需 colbert。"""
        return self._encode(list(texts), batch_size=batch_size,
                            return_colbert=True)

    def _encode(self, texts, batch_size, return_colbert):
        return self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=return_colbert,
        )

    def colbert_score(self, q_reps, p_reps):
        """ColBERT MaxSim 分数，逐候选计算（FlagEmbedding 接口为单查询对单文档）。

        q_reps: 查询 encode 输出的 dict 或单个 colbert 数组 (n_q_tokens, dim)
        p_reps: 候选 encode 输出的 dict 或 colbert 数组列表 [(n_tokens, dim), ...]
        返回: (len(p_reps),) float32 分数数组
        """
        if isinstance(q_reps, dict):
            q_reps = q_reps["colbert_vecs"][0]
        if isinstance(p_reps, dict):
            p_reps = p_reps["colbert_vecs"]
        scores = [float(self._model.colbert_score(q_reps, p)) for p in p_reps]
        return np.asarray(scores, dtype="float32")

    @staticmethod
    def sparse_to_milvus(lexical_weights):
        """FlagEmbedding 稀疏输出 {np.int64: np.float32} -> {int: float}。"""
        return {int(k): float(v) for k, v in lexical_weights.items()}
