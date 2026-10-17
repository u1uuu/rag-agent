# rag-agent — 基于 RAG 的文档问答系统

一个**从零手写**的检索增强生成（RAG）问答系统：喂入一批文档，系统能根据文档内容回答你的问题。

> 定位：个人技术转型证据项目。核心逻辑全部手写（切块 → 向量化 → 检索 → 生成），不套 LangChain 全家桶，目标是能讲清每一步的"为什么"。

## 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| 语言 | Python 3.11 | 虚拟环境 `.venv` |
| 大模型 | DeepSeek API | `deepseek-chat`，OpenAI SDK 兼容 |
| 向量化 | BGE 中文模型 | 第 2 周接入 |
| 检索 | numpy 余弦相似度 | 手写，不依赖向量库 |
| 配置 | python-dotenv | `.env` 存 key（已 gitignore） |

## 目录结构

```
rag-agent/
├── data/        # 喂入的文档（论文/行业资料/公司制度）
├── src/         # 核心代码
│   ├── chunker.py    # 文档切块
│   ├── embed.py      # 向量化
│   ├── retrieve.py   # 检索
│   ├── generate.py   # 生成答案
│   └── main.py       # 问答入口
├── scripts/     # 工具脚本（如 test_api.py 连接验证）
├── eval/        # 50 题评测集与结果
├── .env         # API key（不入库）
└── .env.example # 环境变量模板
```

## 快速开始

```bash
# 1. 安装依赖（已建好 .venv）
.venv\Scripts\python -m pip install -r requirements.txt

# 2. 配置 key
#    复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY

# 3. 验证连接
.venv\Scripts\python scripts\test_api.py

# 4. 运行问答（第 2 周可用）
.venv\Scripts\python src\main.py
```

## 评测（第 3 周补充）

计划用 50 个问答对测试，记录答对率、错误类型与修复过程。
