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

## 评测（真实结果）

用 **50 个问答对**（取自 3 篇真实文档：公司制度 / 学术研究 / 行业分析，答案都能在文档中找到）自动评测。

**自动调参**（`eval/tune.py` 遍历 6 组参数组合，每组跑 50 题）：

| 切块上限 | 检索 3 块 | 检索 5 块 |
|---|---|---|
| 200 字 | 94% | 98% |
| **250 字** | 90% | **100% ★最优** |
| 300 字 | 94% | 98% |

**结论**：检索 5 块全面优于 3 块；单块上限 250 字 + 检索 5 块为最优配置。已固化到主代码（`chunker.py` 默认 250 字、`main.py` 检索 5 块）。

**调优过程中的关键发现**（都是真实踩坑）：
1. 单块 500 字时，长文档整块向量被"平均"，检索分被稀释 → 调 250 字
2. 检索块数太少会漏召回，模型看漏答案 → 3 块提到 5 块
3. 参数是"组合"不是单点：切块 250 字时配 top_k=3 反而最差（90%），配 top_k=5 才最优（100%）

**错误分析**：剩余错误主要是"模型拒绝推断"——文档只有间接表述、无直接答案时，模型按提示词要求诚实回答"资料中没有"，而非编造（防幻觉机制生效）。

脚本：`eval/run_eval.py`（评测）、`eval/tune.py`（自动调参）、`eval/questions.json`（50 题）、`eval/results.json`（结果）。

