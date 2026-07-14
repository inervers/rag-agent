# RAG Agent 学习

从 Function Calling 到 RAG API 服务的 Agent 技术学习记录。

## 目录

| Day | 主题 | 内容 |
|---|---|---|
| **Day1** | Function Calling 基础 | 注册工具（计算器、天气、知识搜索），LLM 决定是否调用、调哪个、传什么参数，执行结果回传后组织回答 |
| **Day2** | 固定管道 RAG API | 用 FastAPI 包装固定检索→生成管道，LCEL 链式调用，GET /health + POST /query |
| **Day3** | FC RAG API | Function Calling 驱动 RAG：LLM 自己判断是否需要搜索、用什么搜索词、可多次搜索 |

## 两种 RAG 架构对比

| | 固定管道 (Day2) | FC 驱动 (Day3) |
|---|---|---|
| 检索时机 | 每次必搜 | LLM 判断 |
| 搜索词 | 用户原句 | LLM 生成的搜索词 |
| 多次搜索 | 不支持 | 支持，可调整搜索策略 |
| 闲聊 | 也搜一次 | 不搜 |
| 代码量 | 简洁（1条 LCEL 链） | 稍多（FC 循环） |

## 快速开始

```powershell
# 设置 API Key
$env:DEEPSEEK_API_KEY="sk-xxx"

# 启动 FC RAG API（Day3）
cd day03_rag_api_fc
python rag_api.py

# 启动固定管道 RAG API（Day2）
cd day02_rag_api_fixed
python rag_api.py
```

### 测试接口

```powershell
# 健康检查
curl http://localhost:8000/health

# 问答
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"Who created Python?"}'
```

### 运行 FC 演示（Day1）

```powershell
cd day01_function_calling
python fc_demo.py
```

## 环境依赖

- Python 3.10+
- fastapi + uvicorn
- langchain-core / langchain-text-splitters
- httpx
- transformers + torch + numpy
- sentence-transformers/all-MiniLM-L6-v2（自动缓存）
