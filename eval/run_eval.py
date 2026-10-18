"""50 题评测脚本（第 3 周核心交付）
对评测集里的每个问题跑完整 RAG（切块->向量化->检索->生成），
再用 DeepSeek 判定模型答案是否命中标准答案要点，
最后统计答对率、输出每题的检索分和错误明细。

用法: .venv\\Scripts\\python eval\\run_eval.py
结果: 打印汇总 + 写入 eval/results.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.chunker import split_into_chunks
from src.embed import embed
from src.generate import generate
from src.retrieve import retrieve

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
Q_FILE = os.path.join(os.path.dirname(__file__), "questions.json")
R_FILE = os.path.join(os.path.dirname(__file__), "results.json")

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    return _client


def build_index():
    """读 data/ 所有非空 txt -> 切块 -> 向量化。"""
    docs = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.lower().endswith(".txt"):
            continue
        p = os.path.join(DATA_DIR, fn)
        if os.path.getsize(p) == 0:
            print(f"  ⚠ 跳过空文件: {fn}")
            continue
        with open(p, encoding="utf-8") as f:
            docs.append(f.read())
        print(f"  读取文档: {fn}")
    chunks = []
    for d in docs:
        chunks.extend(split_into_chunks(d, max_chars=250))
    print(f"  共切 {len(chunks)} 块，正在向量化...")
    vecs = embed(chunks)
    return chunks, vecs


def judge(question: str, ref: str, answer: str) -> dict:
    """用 DeepSeek 判定模型答案是否正确命中标准答案要点。"""
    prompt = (
        "判断下面的模型回答是否正确回答了问题、是否包含标准答案的关键要点。\n"
        f"问题：{question}\n"
        f"标准答案要点：{ref}\n"
        f"模型回答：{answer}\n"
        '只输出 JSON：{"correct": true或false, "reason": "一句话理由"}'
    )
    resp = _get_client().chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"correct": False, "reason": "判定解析失败"}


def main() -> None:
    with open(Q_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"评测集: {len(questions)} 题\n")

    chunks, vecs = build_index()
    if not chunks:
        print("❌ data/ 里没有可用的文档，无法评测。")
        sys.exit(1)

    correct = 0
    details = []
    for q in questions:
        q_text = q["question"]
        qv = embed([q_text])[0]
        hits = retrieve(qv, vecs, chunks, top_k=5)
        top_score = round(hits[0][1], 3) if hits else 0.0
        answer = generate(q_text, [c for c, _ in hits])
        verdict = judge(q_text, q["answer"], answer)
        ok = bool(verdict.get("correct"))
        correct += ok
        details.append({
            "id": q["id"], "source": q.get("source", ""),
            "question": q_text, "answer": q["answer"],
            "model_answer": answer,
            "retrieval_top_score": top_score,
            "correct": ok, "reason": verdict.get("reason", ""),
        })
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] #{q['id']} 检索分={top_score:.3f} | {q_text[:30]}")

    accuracy = correct / len(questions)
    print(f"\n========== 评测结果 ==========")
    print(f"总题数: {len(questions)}  答对: {correct}  答错: {len(questions) - correct}")
    print(f"答对率: {accuracy * 100:.1f}%")

    wrong = [d for d in details if not d["correct"]]
    print(f"\n--- 错误明细（{len(wrong)} 题）---")
    for d in wrong:
        print(f"#{d['id']} 检索分={d['retrieval_top_score']:.3f}")
        print(f"  问: {d['question']}")
        print(f"  答: {d['model_answer']}")
        print(f"  判定理由: {d['reason']}\n")

    with open(R_FILE, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "total": len(questions),
                   "correct": correct, "details": details}, f,
                  ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {R_FILE}")


if __name__ == "__main__":
    main()
