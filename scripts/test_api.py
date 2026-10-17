"""DeepSeek API 连接验证脚本
运行: python scripts/test_api.py
用途: 确认 .env 里的 key 有效、网络通、能拿到模型回复。
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 读取项目根目录的 .env

key = os.getenv("DEEPSEEK_API_KEY", "").strip()
base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not key or key.startswith("sk-在这里"):
    print("❌ .env 里还没有有效 key，请填写后重试")
    raise SystemExit(1)

client = OpenAI(api_key=key, base_url=base)

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "请只回复五个字：连接成功"}],
    max_tokens=20,
    temperature=0,
)

print("✅ 连接成功，模型回复:", resp.choices[0].message.content)
