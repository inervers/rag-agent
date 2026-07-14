# RAG Agent 学习

从 Function Calling 到 RAG API 服务，再到 Multi-Agent 系统的 Agent 技术学习记录。

## 目录

| Day | 主题 | 内容 | 状态 |
|---|---|---|---|
| **Day1** | Function Calling 基础 | 6 种工具（计算器、天气、知识搜索等），多工具并行调用、链式组合 | ✅ |
| **Day2** | 固定管道 RAG API | FastAPI + LCEL 管道，GET /health + POST /query | ✅ |
| **Day3** | FC RAG API（增强版） | Chroma 持久化 + SSE 流式响应 + .env 自动加载 | ✅ |
| **Day4** | 架构对比评测 | 固定管道 vs FC 三项指标量化对比 | ✅ |
| **Day5** | Multi-Agent 入门 | Supervisor + Worker 模式，多专业 Agent 协作 | ✅ |

## 各 Day 详解

### Day1 — Function Calling 基础

`day01_function_calling/fc_demo.py`

6 个演示工具：计算器、天气查询、知识搜索、添加笔记、发送邮件、当前时间。
覆盖 5 种场景：单工具调用、并行多工具、先加后查、链式组合、并行混合。

### Day2 — 固定管道 RAG API

`day02_rag_api_fixed/rag_api.py`

固定检索→生成管道（每次必搜），LCEL 表达式链式调用。特点是速度快（~1s/次）、答案一致性好，但缺乏灵活性。

**接口：** `POST /query` `/doc` `GET /health`

### Day3 — FC RAG API（Chroma 持久化 + SSE 流式）

`day03_rag_api_fc/rag_api.py`

四个工具：search_knowledge / add_document / summarize / translate。

增强功能：
- **Chroma 持久化** — 知识库保存到 `chroma_db/` 目录，重启服务数据不丢
- **SSE 流式响应** — `POST /query/stream` 逐 token 返回，支持打字机效果
- **.env 自动加载** — 从 `OH-WorkSpace/.env` 自动读取 API Key，启动无需手动设置
- **FC 中间轮日志** — 每次调工具打印 🛠 日志，链式调用清晰可见

### Day4 — 固定管道 vs FC 对比评测

`day04_evaluation/eval_compare.py`

5 道测试题，三项指标量化对比：

| 指标 | 固定管道 | FC | 差异 |
|---|---|---|---|
| 覆盖度 | **0.92** | 0.90 | FC 略低 |
| 忠实度 | **1.0** | **1.0** | 持平 |
| 平均耗时 | **1.02s** | 4.15s | FC 慢 4x |

**结论：** 固定管道适合确定性高、追求速度的场景（客服 FAQ）；FC 适合需要多步推理、灵活决策的场景（搜索→总结→翻译链），代价是速度。

### Day5 — Multi-Agent 入门

`day05_multi_agent/multi_agent.py`

Supervisor + Workers 模式，4 个 Agent 各司其职：

```
用户提问
    │
    ▼
 Supervisor Agent     ← 判断需求，路由到专业 Agent
    │
    ├──→ Retrieval Agent     ← 专职搜索/添加知识库
    ├──→ Text Agent          ← 专职总结/翻译
    └──→ Writer Agent        ← 专职润色排版
```

每个 Agent 有自己的系统提示词、工具集和消息历史，Supervisor 通过工具调用调度它们。
演示 4 个场景：直接问答、检索→写作、检索→总结→写作、检索→翻译→写作。

## 快速开始

### 安装依赖

```powershell
pip install --user chromadb fastapi uvicorn httpx transformers torch numpy langchain-core langchain-text-splitters
```

### 启动 FC RAG API（Day3 — 推荐）

```powershell
cd day03_rag_api_fc
python rag_api.py
```

### 测试接口

```powershell
# 健康检查
curl http://localhost:8000/health

# 非流式问答
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"What is Python?"}'

# 流式问答（逐字返回）
curl -N -X POST http://localhost:8000/query/stream -H "Content-Type: application/json" -d '{"question":"What is PyTorch?"}'

# 动态添加知识
curl -X POST http://localhost:8000/doc -H "Content-Type: application/json" -d '{"title":"FastAPI","content":"FastAPI is a modern Python web framework for building APIs."}'
```

### 运行其他演示

```powershell
# Day1 — FC 基础演示
cd day01_function_calling
python fc_demo.py

# Day4 — 架构对比评测
cd day04_evaluation
python eval_compare.py

# Day5 — Multi-Agent 演示
cd day05_multi_agent
python multi_agent.py
```

### API Key

脚本会自动从 `OH-WorkSpace/.env` 文件读取 `DEEPSEEK_API_KEY`。只需在 `C:\Users\inervers\Desktop\OH-WorkSpace\.env` 中写入：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

## 环境依赖

- Python 3.10+
- chromadb（Day3/Day5 知识库持久化）
- fastapi + uvicorn
- langchain-core / langchain-text-splitters
- httpx（DeepSeek API 调用）
- transformers + torch + numpy
- sentence-transformers/all-MiniLM-L6-v2（向量嵌入，自动缓存）
