"""RAG 文档问答入口（向量检索版，主线）
用法：
  1. 把文档（.txt）放进 data/
  2. 运行：.venv\\Scripts\\python src\\main.py
  3. 输入问题回车，Ctrl+C 退出

流程：读文档 -> 切块 -> BGE向量化 -> 余弦检索 -> DeepSeek生成
"""
import os
import sys

from dotenv import load_dotenv

from src.chunker import split_into_chunks
from src.embed import embed
from src.generate import generate
from src.retrieve import retrieve

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_docs(data_dir: str) -> list[str]:
    """读取 data/ 下所有 .txt 文档。"""
    docs = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.lower().endswith(".txt"):
            with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
                docs.append(f.read())
            print(f"  读取文档: {fn}")
    if not docs:
        print("⚠  data/ 目录下没有 .txt 文档，请先放入文档。")
    return docs


def build_index(data_dir: str):
    """读文档 -> 切块 -> 向量化，返回 (chunks, chunk_vecs)。"""
    docs = load_docs(data_dir)
    chunks = []
    for doc in docs:
        chunks.extend(split_into_chunks(doc))
    print(f"  切块完成，共 {len(chunks)} 块")
    print("  正在向量化（首次会加载本地 BGE 模型，稍等）...")
    chunk_vecs = embed(chunks)
    print(f"  向量化完成，索引就绪。\n")
    return chunks, chunk_vecs


def main() -> None:
    chunks, chunk_vecs = build_index(DATA_DIR)
    if not chunks:
        sys.exit(1)

    print("=" * 50)
    print("RAG 问答就绪。输入问题回车；输入 q 退出。")
    print("=" * 50)
    while True:
        q = input("\n问题: ").strip()
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            print("再见。")
            break
        q_vec = embed([q])[0]
        hits = retrieve(q_vec, chunk_vecs, chunks, top_k=5)
        contexts = [c for c, _ in hits]
        answer = generate(q, contexts)
        print(f"\n答案: {answer}")


if __name__ == "__main__":
    main()
