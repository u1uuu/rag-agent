"""端到端验证 RAG 最小版：读文档 -> 切块 -> 向量化 -> 检索 -> 问答
逐段打印中间结果，方便你确认每个环节都正常。
用法: .venv\\Scripts\\python scripts\\test_rag.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.chunker import split_into_chunks
from src.embed import embed
from src.generate import generate
from src.retrieve import retrieve

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "demo_行业资料.txt")
QUESTION = "澳大利亚 482 签证需要满足什么基本条件？"


def main() -> None:
    with open(DATA, encoding="utf-8") as f:
        text = f.read()

    chunks = split_into_chunks(text)
    print(f"[1] 切块: 全文 {len(text)} 字 -> {len(chunks)} 块")
    for i, c in enumerate(chunks):
        print(f"    块{i}: {len(c)}字 | {c[:30]}...")

    print("[2] 向量化（首次加载 BGE 模型）...")
    vecs = embed(chunks)
    print(f"    完成: {len(vecs)} 个向量，维度 {len(vecs[0])}")

    print(f"[3] 检索问题: {QUESTION}")
    qv = embed([QUESTION])[0]
    hits = retrieve(qv, vecs, chunks, top_k=3)
    for c, s in hits:
        print(f"    score={s:.3f} | {c[:45]}...")

    print("[4] DeepSeek 生成答案...")
    answer = generate(QUESTION, [c for c, _ in hits])
    print(f"\n【答案】{answer}")


if __name__ == "__main__":
    main()
