"""生成模块
把检索到的相关块拼进提示词，交给 DeepSeek 生成答案。

关键约束（提示词里写死）：
- 只依据提供的资料回答，资料里没有的必须说"资料中没有"，禁止编造
- 这既是正确性保证，也是面试时能讲清的"防幻觉"手段
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    return _client


def generate(question: str, contexts: list[str]) -> str:
    """question + 相关文档块 -> 模型答案。"""
    ctx_text = "\n\n---\n\n".join(contexts)
    resp = get_client().chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system",
             "content": "你是文档问答助手。只依据用户提供的资料回答问题；"
                        "资料中没有的信息，明确回答'资料中没有相关内容'，绝对不要编造。"},
            {"role": "user",
             "content": f"以下是相关资料：\n\n{ctx_text}\n\n"
                        f"问题：{question}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
