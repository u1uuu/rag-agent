# ===== 最小 RAG：读文档 -> 切块 -> 检索 -> 问 DeepSeek =====
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# 1. 读取 .env 里的 DeepSeek key
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 2. 读 data 文件夹里所有 txt
def read_docs(folder="data"):
    docs = []
    for path in Path(folder).glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        docs.append({"name": path.name, "text": text})
    return docs

# 3. 切块：每块 chunk_size 个字，相邻块重叠 overlap 个字
def chunk_docs(docs, chunk_size=500, overlap=50):
    chunks = []
    for doc in docs:
        text = doc["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end]
            chunks.append({"source": doc["name"], "text": piece})
            start = end - overlap  # 往回退 overlap 个字，避免一句话被切断
    return chunks

# 4. 超简单检索：问题里的字，在块里出现得越多，分越高
def retrieve(question, chunks, top_k=3):
    scored = []
    for c in chunks:
        score = sum(1 for ch in question if ch in c["text"])
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored[:top_k]]

# 5. 拼 prompt，问模型
def ask(question, top_chunks):
    context = "\n\n".join(
        f"[来源:{c['source']}]\n{c['text']}" for c in top_chunks
    )
    prompt = f"""你只能根据下面的材料回答问题。
如果材料里没有答案，就回答“材料里没有找到”。
不要编造。

材料：
{context}

问题：{question}
"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content

# 6. 主流程
if __name__ == "__main__":
    docs = read_docs("data")
    print("读到的文档：", [d["name"] for d in docs])

    chunks = chunk_docs(docs, chunk_size=200, overlap=50)
    print("切成了多少块：", len(chunks))

    question = input("请输入你的问题：")
    top = retrieve(question, chunks, top_k=3)

    print("\n===== 检索到的片段 =====")
    for c in top:
        print(f"来源 {c['source']}：{c['text'][:100]}...")

    print("\n===== 模型回答 =====")
    print(ask(question, top))