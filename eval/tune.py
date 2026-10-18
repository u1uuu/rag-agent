"""自动调参脚本：自动试不同参数组合，跑 50 题评测，找出最优切块/检索参数。
这就是工程上的"超参数搜索"——不需要手动一次次试，代码自动遍历。

用法: .venv\\Scripts\\python eval\\tune.py
默认遍历: max_chars(单块上限) x top_k(检索块数)，输出每个组合的答对率，标出最优。
"""
import itertools
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

_cli = None


def get_openai():
    global _cli
    if _cli is None:
        _cli = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    return _cli


def build_index(max_chars: int, overlap: int):
    docs = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.lower().endswith(".txt"):
            continue
        p = os.path.join(DATA_DIR, fn)
        if os.path.getsize(p) == 0:
            continue
        with open(p, encoding="utf-8") as f:
            docs.append(f.read())
    chunks = []
    for d in docs:
        chunks.extend(split_into_chunks(d, max_chars=max_chars, overlap=overlap))
    vecs = embed(chunks)
    return chunks, vecs


def judge(question, ref, answer):
    prompt = (
        "判断下面的模型回答是否正确回答了问题、是否包含标准答案的关键要点。\n"
        f"问题：{question}\n标准答案要点：{ref}\n模型回答：{answer}\n"
        '只输出 JSON：{"correct": true或false, "reason": "一句话理由"}'
    )
    resp = get_openai().chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"correct": False, "reason": "判定解析失败"}


def run_combo(max_chars: int, top_k: int, overlap: int) -> tuple:
    chunks, vecs = build_index(max_chars, overlap)
    with open(Q_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    correct = 0
    for q in questions:
        qv = embed([q["question"]])[0]
        hits = retrieve(qv, vecs, chunks, top_k=top_k)
        answer = generate(q["question"], [c for c, _ in hits])
        verdict = judge(q["question"], q["answer"], answer)
        if verdict.get("correct"):
            correct += 1
    return correct, len(questions)


def main():
    # 待尝试的参数组合（可自由增删）
    max_chars_list = [200, 250, 300]
    top_k_list = [3, 5]
    overlap = 50

    results = []
    for max_chars, top_k in itertools.product(max_chars_list, top_k_list):
        label = f"max_chars={max_chars}, top_k={top_k}"
        print(f"▶ 正在测: {label}")
        try:
            correct, total = run_combo(max_chars, top_k, overlap)
            acc = correct / total
            results.append((acc, max_chars, top_k, correct, total))
            print(f"   完成: 答对 {correct}/{total} = {acc * 100:.1f}%\n")
        except Exception as e:
            print(f"   失败: {e}\n")

    print("=" * 40)
    print("自动调参结果（按答对率排序）")
    for acc, max_chars, top_k, correct, total in sorted(results, reverse=True):
        mark = " ★最优" if acc == max(r for r, *_ in results) else ""
        print(f"  max_chars={max_chars:>3}  top_k={top_k}  答对率={acc * 100:.1f}%  ({correct}/{total}){mark}")


if __name__ == "__main__":
    main()
