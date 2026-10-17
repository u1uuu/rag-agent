"""检索模块
用余弦相似度（向量点积）找出与问题最相关的 top-k 个文档块。

手写，不依赖向量数据库，方便讲清楚原理：
- 把所有块向量拼成矩阵，问题向量与之做点积，得到每个块的相关性分数
- 取分数最高的 top_k 块作为"上下文"喂给模型
"""
import numpy as np


def retrieve(query_vec: list[float], chunk_vecs: list[list[float]],
             chunks: list[str], top_k: int = 3):
    """返回 [(chunk, score), ...]，按相关度从高到低，最多 top_k 个。"""
    if not chunk_vecs:
        return []
    matrix = np.array(chunk_vecs, dtype=np.float32)
    q = np.array(query_vec, dtype=np.float32)
    scores = matrix @ q                      # 归一化后点积 = 余弦相似度
    idx = np.argsort(scores)[::-1][:top_k]   # 从高到低取 top_k
    return [(chunks[i], float(scores[i])) for i in idx]
