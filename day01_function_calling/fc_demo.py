"""Function Calling 演示 Day1：多工具组合版

6 个工具，展示 LLM 如何组合多个工具完成复合任务：
  - calculator     : 数学计算（已有的）
  - get_weather    : 天气查询（已有的）
  - search_knowledge : 知识库搜索（已有的）
  - add_document   : 动态添加新知识（新增）
  - summarize      : 文本摘要（新增）
  - translate      : 翻译（新增）

关键看点：LLM 可以串行链式调用（搜→摘要→翻译）和并行多工具调用。
"""

import sys, os, json, math

_REAL_USER_SITE = r"C:\Users\inervers\AppData\Roaming\Python\Python313\site-packages"
if os.path.isdir(_REAL_USER_SITE) and _REAL_USER_SITE not in sys.path:
    sys.path.insert(0, _REAL_USER_SITE)

import httpx

# 自动搜索 .env：从脚本目录往上找，直到 OH-WorkSpace 根目录
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    search_dir = os.path.dirname(__file__)
    for _ in range(6):
        env_path = os.path.join(search_dir, ".env")
        if os.path.isfile(env_path):
            for line in open(env_path, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "DEEPSEEK_API_KEY":
                        DEEPSEEK_API_KEY = v.strip()
            break
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
if not DEEPSEEK_API_KEY:
    print("需要设置 DeepSeek API Key，在 .env 文件中写入 DEEPSEEK_API_KEY=sk-xxx")
    exit(1)

# =============================================
# 1. 定义工具
# =============================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "数学计算，支持 +, -, *, /, **, sqrt, sin, cos",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "表达式如 '3.14 * 2'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 'Beijing'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索内部知识库。知识库包含 Python、PyTorch、Transformer、RAG、Chroma、LangChain 等主题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_document",
            "description": "向知识库添加一条新的知识条目",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "知识条目标题"},
                    "content": {"type": "string", "description": "知识条目的具体内容"},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "对一段文字进行摘要总结",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "需要摘要的文本"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "将文本翻译为目标语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "需要翻译的文本"},
                    "target_language": {"type": "string", "description": "目标语言，如 Chinese, Japanese, English, French"},
                },
                "required": ["text", "target_language"],
            },
        },
    },
]

# =============================================
# 2. 工具实现
# =============================================

# --- 工具间共享的知识库 ---
_knowledge_base = [
    "Python was created by Guido van Rossum in 1991.",
    "PyTorch was developed by Meta AI. It provides GPU-accelerated tensor computation.",
    "The Transformer architecture was introduced by Google in 2017. It uses self-attention.",
    "RAG combines a retriever with a generator. The retriever finds relevant documents.",
    "Chroma is an open-source vector database for AI applications.",
    "LangChain is a framework for developing LLM applications with chains and agents.",
]

def tool_calculator(expression: str) -> str:
    allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
               "pi": math.pi, "e": math.e}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

def tool_get_weather(city: str) -> str:
    weather_data = {
        "Beijing": "26°C, 晴", "北京": "26°C, 晴",
        "Shanghai": "30°C, 多云", "上海": "30°C, 多云",
        "Tokyo": "28°C, 小雨", "东京": "28°C, 小雨",
        "London": "18°C, 阴", "伦敦": "18°C, 阴",
        "New York": "25°C, 晴间多云", "纽约": "25°C, 晴间多云",
    }
    result = weather_data.get(city, f"没有 {city} 的天气数据")
    return f"{city}：{result}"

def tool_search_knowledge(query: str) -> str:
    results = [doc for doc in _knowledge_base if query.lower() in doc.lower()]
    return "\n".join(results) if results else f"未找到与 '{query}' 相关的信息"

def tool_add_document(title: str, content: str) -> str:
    _knowledge_base.append(f"{title}：{content}")
    return f"成功添加文档「{title}」，当前知识库共 {len(_knowledge_base)} 条"

def tool_summarize(text: str) -> str:
    """用 LLM 做摘要（复用 DeepSeek）"""
    body = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "You are a summarizer. Provide a concise summary in the same language as the text."},
            {"role": "user", "content": f"Summarize this:\n\n{text}"},
        ],
        "temperature": 0.2,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    resp = httpx.post(
        "https://api.deepseek.com/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def tool_translate(text: str, target_language: str) -> str:
    """用 LLM 做翻译（复用 DeepSeek）"""
    body = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": f"You are a professional translator. Translate the text to {target_language}. Output only the translation."},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    resp = httpx.post(
        "https://api.deepseek.com/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

TOOL_IMPLS = {
    "calculator": tool_calculator,
    "get_weather": tool_get_weather,
    "search_knowledge": tool_search_knowledge,
    "add_document": tool_add_document,
    "summarize": tool_summarize,
    "translate": tool_translate,
}

# =============================================
# 3. LLM 调用
# =============================================

client = httpx.Client(timeout=30)

def call_llm(messages, tools=None):
    body = {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.3,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    if tools:
        body["tools"] = tools

    resp = client.post(
        "https://api.deepseek.com/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]

# =============================================
# 4. FC 循环
# =============================================

def run_with_tools(user_query: str, max_rounds=8):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. "
         "Use them when needed. You can call multiple tools in parallel. "
         "Answer in the same language as the user."},
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*55}")
    print(f"  用户：{user_query}")

    for turn in range(max_rounds):
        msg = call_llm(messages, tools=TOOLS)

        if not msg.get("tool_calls"):
            print(f"  🤖 回答：{msg['content']}\n")
            return msg["content"]

        assistant_msg = {
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": msg["tool_calls"],
        }
        messages.append(assistant_msg)

        for tool_call in msg["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            print(f"  🛠  调用工具：{func_name}({func_args})")

            impl = TOOL_IMPLS.get(func_name)
            if impl:
                result = impl(**func_args)
            else:
                result = f"未知工具：{func_name}"

            result_preview = result[:80] + "..." if len(result) > 80 else result
            print(f"     → {result_preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(result),
            })

    print("  ⚠️  达到最大轮数")
    return messages[-1]["content"]

# =============================================
# 5. 演示场景
# =============================================

print("=" * 55)
print("Function Calling 演示——多工具组合")
print("=" * 55)

# ── 场景 1：并行调用 ──
run_with_tools("同时查一下北京和上海的天气")

# ── 场景 2：动态加知识 ──
run_with_tools("我想加一条知识：Kubernetes 是一个容器编排平台，由 Google 开源。然后搜一下看看能搜到吗？")

# ── 场景 3：链式组合（搜→摘要→翻译） ──
run_with_tools("搜索关于 Transformer 的知识，然后用自己的话总结一遍，再翻译成中文")

# ── 场景 4：计算 + 搜索 并行 ──
run_with_tools("计算 256 的平方根是多少？同时查一下 PyTorch 的相关知识。")

# ── 场景 5：复合场景 ──
run_with_tools("先加一条知识：FastAPI 是一个高性能 Python Web 框架，支持异步。然后查一下 FastAPI 能看到什么结果？")
