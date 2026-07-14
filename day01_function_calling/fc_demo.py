"""L4-Day2: Function Calling（工具调用）

核心模式：
  用户提问 → LLM 决定调哪个工具 → 执行工具 → 结果给 LLM → LLM 组织最终回答

DeepSeek 支持 OpenAI 兼容的 function calling 协议。
"""

import sys, os, json, math

_REAL_USER_SITE = r"C:\Users\inervers\AppData\Roaming\Python\Python313\site-packages"
if os.path.isdir(_REAL_USER_SITE) and _REAL_USER_SITE not in sys.path:
    sys.path.insert(0, _REAL_USER_SITE)

import httpx

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    key = input("请输入 DeepSeek API Key: ").strip()
    if key:
        DEEPSEEK_API_KEY = key
    else:
        exit(1)

# =============================================
# 1. 定义工具（JSON Schema 格式）
# =============================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式的值。支持 +, -, *, /, **, sqrt, sin, cos",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '3.14 * 2' 或 'sqrt(144)'",
                    }
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
                    "city": {
                        "type": "string",
                        "description": "城市名，如 'Beijing', 'Shanghai', 'Tokyo'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索内部知识库，查找与问题相关的信息。知识库包含 Python、PyTorch、Transformer、RAG、Chroma、LangChain 等主题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# =============================================
# 2. 工具的具体实现
# =============================================

def tool_calculator(expression: str) -> str:
    """执行数学计算"""
    allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
               "pi": math.pi, "e": math.e}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

def tool_get_weather(city: str) -> str:
    """模拟天气查询（演示用）"""
    weather_data = {
        "Beijing": "26°C, 晴",
        "Shanghai": "30°C, 多云",
        "Tokyo": "28°C, 小雨",
        "London": "18°C, 阴",
        "New York": "25°C, 晴间多云",
    }
    result = weather_data.get(city, f"没有 {city} 的天气数据")
    return f"{city}：{result}"

def tool_search_knowledge(query: str) -> str:
    """搜索内部知识库"""
    kb = [
        "Python was created by Guido van Rossum in 1991.",
        "PyTorch was developed by Meta AI. It provides GPU-accelerated tensor computation.",
        "The Transformer architecture was introduced by Google in 2017. It uses self-attention.",
        "RAG combines a retriever with a generator. The retriever finds relevant documents.",
        "Chroma is an open-source vector database for AI applications.",
        "LangChain is a framework for developing LLM applications with chains and agents.",
    ]
    # 简单关键词匹配
    results = [doc for doc in kb if query.lower() in doc.lower()]
    return "\n".join(results) if results else f"未找到与 '{query}' 相关的信息"

# 工具名称 → 实现函数 映射
TOOL_IMPLS = {
    "calculator": tool_calculator,
    "get_weather": tool_get_weather,
    "search_knowledge": tool_search_knowledge,
}

# =============================================
# 3. LLM 调用
# =============================================

client = httpx.Client(timeout=30)

def call_llm(messages, tools=None):
    """发起一次 LLM 调用，可选带 tools"""
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
# 4. Function Calling 循环
# =============================================

def run_with_tools(user_query: str, max_rounds=5):
    """
    Function Calling 主循环：
    1. 用户提问
    2. LLM 决定要不要调工具
    3. 如果要调，解析工具名和参数 → 执行 → 结果送回 LLM
    4. LLM 用工具结果组织最终回答
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. "
         "Use them when needed. Answer in the same language as the user."},
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'=' * 55}")
    print(f"  用户：{user_query}")

    for turn in range(max_rounds):
        msg = call_llm(messages, tools=TOOLS)

        # ── 情况 A：LLM 决定直接回答（没有 tool_calls） ──
        if not msg.get("tool_calls"):
            print(f"  🤖 回答：{msg['content']}\n")
            return msg["content"]

        # ── 情况 B：LLM 想调工具 ──
        # 先把 LLM 的 tool_calls 响应加入消息列表（只加一次）
        assistant_msg = {"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]}
        messages.append(assistant_msg)

        for tool_call in msg["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            print(f"  🛠  调用工具：{func_name}({func_args})")

            # 执行工具
            impl = TOOL_IMPLS.get(func_name)
            if impl:
                result = impl(**func_args)
            else:
                result = f"未知工具：{func_name}"

            print(f"     → 结果：{result}")

            # 工具执行结果送回 LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(result),
            })

    print("  ⚠️  达到最大轮数")
    return messages[-1]["content"]

# =============================================
# 5. 演示
# =============================================

print("=" * 55)
print("Function Calling 演示")
print("=" * 55)

# ── 场景 1：需要计算器 ──
run_with_tools("计算 3.14 的平方乘以 2 等于多少？")

# ── 场景 2：需要查天气 ──
run_with_tools("北京和东京的天气怎么样？")

# ── 场景 3：需要搜索知识库 ──
run_with_tools("PyTorch 是哪个公司开发的？它有什么特点？")

# ── 场景 4：不需要工具（纯 LLM 知识） ──
run_with_tools("解释一下什么是神经网络？")

# ── 场景 5：多步骤调用（先搜索再回答） ──
run_with_tools("LangChain 和 Chroma 有什么关系？请先搜索知识库再回答。")
