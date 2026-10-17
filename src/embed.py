"""向量化模块
用 BGE 中文模型把文本转成向量。模型下载到本地 models/ 目录，离线可用。

说明：DeepSeek 没有 embedding 接口，所以向量化用本地免费模型 BGE。
- 模型：BAAI/bge-small-zh-v1.5（中文效果好、体积小，约 100MB）
- 首次运行前需要下载模型：见 scripts/download_model.py
"""
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# 模型路径：优先用本地下载的，找不到再退回模型名
_LOCAL_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-zh-v1.5")
_MODEL_REF = os.getenv("EMBEDDING_MODEL", "")

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        ref = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else (_MODEL_REF or "BAAI/bge-small-zh-v1.5")
        _model = SentenceTransformer(ref)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """文本列表 -> 归一化向量列表（归一化后点积=余弦相似度）。"""
    vecs = get_model().encode(texts, normalize_embeddings=True)
    return vecs.tolist()
