"""文档切块模块
核心问题：怎么把长文档切成"适合检索+喂给模型"的块？

切块策略（第 3 周评测时用 50 题验证、调参，这里给第一版）：
1. 先按空行切段落，保留自然语义边界
2. 段落过长时，按中文句号/问号/感叹号切句子，再合并到接近 max_chars
3. 用 overlap 保留块尾几个字，避免"一句话被硬切断"导致语义丢失

为什么这么切：
- 太粗（整篇一个块）：检索时命中不精确，把整篇都塞给模型，浪费 token 且答不准
- 太细（每句一块）：上下文丢失，模型找不到"这个句子在讲谁/什么"
- 500 字左右 + 50 字重叠：是文档问答里常用的平衡点，兼顾精度和上下文
"""
import re

# 默认参数（后续可调）
DEFAULT_MAX_CHARS = 250   # 单块最大字符数（自动调参最优：见 eval/tune.py）
DEFAULT_OVERLAP = 50      # 相邻块重叠字符数


def _split_paragraph(text: str) -> list[str]:
    """按空行（\n\n）切成自然段落。"""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_long_paragraph(para: str, max_chars: int, overlap: int) -> list[str]:
    """把过长段落按句子切分并合并到接近 max_chars，块尾带重叠。"""
    sentences = [s for s in re.split(r"(?<=[。！？；!?])", para) if s.strip()]
    chunks = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) > max_chars:
            chunks.append(buf)
            buf = buf[-overlap:] + s  # 新块带上旧块尾部重叠
        else:
            buf += s
    if buf:
        chunks.append(buf)
    return chunks


def split_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHARS,
                      overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """把整篇文本切成检索用的块列表。"""
    chunks = []
    for para in _split_paragraph(text):
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            chunks.extend(_split_long_paragraph(para, max_chars, overlap))
    return chunks
