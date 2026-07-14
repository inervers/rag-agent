"""Multi-Agent 系统入门（Supervisor + Workers 模式）

架构：
  用户 → Supervisor Agent → 路由到专业 Agent → 返回结果

Agent 角色：
  - Supervisor  : 接收问题，决定交给哪个 Agent，协调多步任务
  - Retrieval   : 搜索/添加知识库（唯一接触 Chroma 的 Agent）
  - Text        : 文本处理（总结、翻译）
  - Writer      : 根据上下文生成最终回答

每个 Agent 独立维护自己的 system prompt 和消息历史。
"""

import sys, os, json, time

_REAL_USER_SITE = r"C:\Users\inervers\AppData\Roaming\Python\Python313\site-packages"
if os.path.isdir(_REAL_USER_SITE) and _REAL_USER_SITE not in sys.path:
    sys.path.insert(0, _REAL_USER_SITE)

os.environ.setdefault("HF_HOME", r"C:\Users\inervers\Desktop\OH-WorkSpace\dl-learning\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# 自动搜索 .env
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
                    os.environ.setdefault(k.strip(), v.strip())
            break
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    print("需要设置 DEEPSEEK_API_KEY")
    exit(1)

import numpy as np
import httpx
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.api.types import EmbeddingFunction

# =============================================
# 共享基础设施
# =============================================

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

def embed_texts(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        pooled = model(**inputs).last_hidden_state.mean(dim=1)
    return (pooled / torch.norm(pooled, dim=1, keepdim=True)).numpy()

class MiniLMEmbedding(EmbeddingFunction):
    def __call__(self, texts):
        return embed_texts(texts).tolist()

# Chroma 知识库（复用持久化数据）
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "day03_rag_api_fc", "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=chromadb.config.Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="rag_knowledge", embedding_function=MiniLMEmbedding())
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# LLM 客户端
llm = httpx.Client(timeout=30)

def call_llm(messages, tools=None):
    body = {
        "model": "deepseek-v4-flash", "messages": messages,
        "temperature": 0.3, "thinking": {"type": "disabled"}, "stream": False,
    }
    if tools:
        body["tools"] = tools
    r = llm.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

print("▶ 共享基础设施就绪")
print(f"▶ Chroma 知识库：{collection.count()} 个块")

# =============================================
# Agent 基类
# =============================================

class Agent:
    """一个 Agent = 系统提示词 + 工具定义 + 消息历史 + LLM 调用循环"""
    def __init__(self, name: str, system_prompt: str, tools: list = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.history = [{"role": "system", "content": system_prompt}]

    def reset(self):
        self.history = [{"role": "system", "content": self.system_prompt}]

    def run(self, user_input: str) -> str:
        """执行一次完整的 Agent 推理循环（支持多轮工具调用）"""
        self.history.append({"role": "user", "content": user_input})

        for _ in range(8):
            msg = call_llm(self.history, tools=self.tools if self.tools else None)

            if not msg.get("tool_calls"):
                result = msg["content"]
                self.history.append({"role": "assistant", "content": result})
                return result

            self.history.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
            for tc in msg["tool_calls"]:
                fname = tc["function"]["name"]
                fargs = json.loads(tc["function"]["arguments"] or "{}")
                tool_fn = self._find_tool(fname)
                tool_result = tool_fn(**fargs) if tool_fn else f"未知工具：{fname}"
                print(f"  [{self.name}] 🛠  {fname}({json.dumps(fargs, ensure_ascii=False)})")
                self.history.append({"role": "tool", "tool_call_id": tc["id"], "content": str(tool_result)})

        return self.history[-1].get("content", "")

    def _find_tool(self, name: str):
        for t in self.tools:
            if t["function"]["name"] == name:
                return self.TOOL_IMPLS.get(name)
        return None

    def register_impls(self, impls: dict):
        self.TOOL_IMPLS = impls

# =============================================
# 专业 Agent 定义
# =============================================

# ── Retrieval Agent ──

RETRIEVAL_PROMPT = (
    "You are a Retrieval Specialist. Your only job is to search the knowledge base "
    "and return relevant documents. You have two tools:\n"
    "  search_knowledge(query)  - 搜索知识库\n"
    "  add_document(title, content) - 添加新知识\n\n"
    "Rules:\n"
    "- FOR SEARCH REQUESTS, always use search_knowledge\n"
    "- Return the raw search results without rewriting\n"
    "- If the user asks to add knowledge, use add_document\n"
    "- Answer in the same language as the user."
)

RETRIEVAL_TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "搜索知识库，查找与问题相关的文档",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "add_document",
        "description": "向知识库添加一条新知识",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["title", "content"],
        },
    }},
]

def _retrieval_search(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=3)
    docs = results.get("documents", [[]])[0]
    return "\n".join(docs) if docs else "未找到相关信息"

def _retrieval_add(title: str, content: str) -> str:
    full = f"{title}：{content}"
    chunks = splitter.split_documents([Document(full)])
    start = collection.count() + 1
    ids = [f"doc_{start + i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=[c.page_content for c in chunks], metadatas=[{"source": title} for _ in chunks])
    return f"添加完成，知识库共 {collection.count()} 个块"

retrieval_agent = Agent("Retrieval", RETRIEVAL_PROMPT, RETRIEVAL_TOOLS)
retrieval_agent.register_impls({"search_knowledge": _retrieval_search, "add_document": _retrieval_add})

# ── Text Agent ──

TEXT_PROMPT = (
    "You are a Text Processing Specialist. You handle summarization and translation.\n"
    "  summarize(text)  - 摘要总结\n"
    "  translate(text, target_language) - 翻译\n\n"
    "Rules:\n"
    "- MUST use the tools when asked to summarize or translate\n"
    "- Do not write summaries or translations yourself\n"
    "- Answer in the same language as the user."
)

TEXT_TOOLS = [
    {"type": "function", "function": {
        "name": "summarize",
        "description": "对一段文本进行摘要总结",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "translate",
        "description": "将文本翻译为目标语言",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_language": {"type": "string", "description": "如 Chinese, English, Japanese"},
            },
            "required": ["text", "target_language"],
        },
    }},
]

def _text_summarize(text: str) -> str:
    body = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "You are a summarizer. Provide a concise summary in the same language as the input."},
            {"role": "user", "content": f"Summarize:\n\n{text}"},
        ],
        "temperature": 0.2, "thinking": {"type": "disabled"},
    }
    r = llm.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    return r.json()["choices"][0]["message"]["content"]

def _text_translate(text: str, target_language: str) -> str:
    body = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": f"You are a translator. Translate to {target_language}. Output only the translation."},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2, "thinking": {"type": "disabled"},
    }
    r = llm.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    return r.json()["choices"][0]["message"]["content"]

text_agent = Agent("Text", TEXT_PROMPT, TEXT_TOOLS)
text_agent.register_impls({"summarize": _text_summarize, "translate": _text_translate})

# ── Writer Agent ──

WRITER_PROMPT = (
    "You are a Writing Specialist. You produce polished, well-structured answers "
    "based on the provided context.\n\n"
    "You do NOT search for information yourself. You only write based on what the "
    "user gives you in the conversation.\n\n"
    "Rules:\n"
    "- Format answers clearly with sections if appropriate\n"
    "- Use the same language as the user\n"
    "- If given search results, cite them as your knowledge source\n"
    "- If you don't have enough information, say so clearly"
)

writer_agent = Agent("Writer", WRITER_PROMPT)

# =============================================
# Supervisor Agent
# =============================================

SUPERVISOR_PROMPT = (
    "You are the Supervisor Agent. You coordinate specialized agents to answer the user's questions.\n\n"
    "Available agents (call as tools):\n"
    "  retrieval_agent(task)  - 搜索知识库或添加知识。输入：具体搜索任务描述\n"
    "  text_agent(task)       - 文本总结或翻译。输入：具体处理任务描述\n"
    "  writer_agent(task)     - 根据上下文撰写最终回答。输入：写作任务描述\n\n"
    "Workflow rules:\n"
    "1. FOR TECHNICAL QUESTIONS: first call retrieval_agent to search, then call writer_agent to format the answer\n"
    "2. FOR SUMMARIZE/TRANSLATE: call text_agent directly\n"
    "3. FOR GREETINGS: answer directly without calling any agent\n"
    "4. FOR CHAINED TASKS: call retrieval_agent → text_agent → writer_agent\n"
    "5. Always pass the full context between agents so downstream agents have what they need\n"
    "6. Answer in the same language as the user"
)

SUPERVISOR_TOOLS = [
    {"type": "function", "function": {
        "name": "retrieval_agent",
        "description": "调用检索 Agent 搜索知识库或添加知识。适合技术类问题",
        "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "发送给检索 Agent 的任务描述"}}, "required": ["task"]},
    }},
    {"type": "function", "function": {
        "name": "text_agent",
        "description": "调用文本 Agent 做总结或翻译",
        "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "发送给文本 Agent 的任务描述"}}, "required": ["task"]},
    }},
    {"type": "function", "function": {
        "name": "writer_agent",
        "description": "调用写作 Agent 根据已有信息撰写答案",
        "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "发送给写作 Agent 的任务描述，包含已有信息"}}, "required": ["task"]},
    }},
]

def _supervisor_call_retrieval(task: str) -> str:
    print(f"  [Supervisor] → 派遣 Retrieval Agent")
    retrieval_agent.reset()
    return retrieval_agent.run(task)

def _supervisor_call_text(task: str) -> str:
    print(f"  [Supervisor] → 派遣 Text Agent")
    text_agent.reset()
    return text_agent.run(task)

def _supervisor_call_writer(task: str) -> str:
    print(f"  [Supervisor] → 派遣 Writer Agent")
    writer_agent.reset()
    return writer_agent.run(task)

supervisor = Agent("Supervisor", SUPERVISOR_PROMPT, SUPERVISOR_TOOLS)
supervisor.register_impls({
    "retrieval_agent": _supervisor_call_retrieval,
    "text_agent": _supervisor_call_text,
    "writer_agent": _supervisor_call_writer,
})

# =============================================
# 交互演示
# =============================================

DEMO_QUESTIONS = [
    "hi",
    "What is PyTorch?",
    "What is RAG? Summarize in Chinese",
    "What is Chroma and LangChain? Then translate the answer to Chinese",
]

print("\n" + "=" * 60)
print("  Multi-Agent 系统演示")
print("=" * 60)

for q in DEMO_QUESTIONS:
    print(f"\n\n>>> 用户：{q}")
    print("-" * 40)
    t0 = time.time()
    answer = supervisor.run(q)
    elapsed = round(time.time() - t0, 2)
    print(f"\n  回答（{elapsed}s）：{answer}")
    supervisor.reset()

    # 在各 Agent 之间也重置
    retrieval_agent.reset()
    text_agent.reset()
    writer_agent.reset()

print("\n" + "=" * 60)
print("  演示结束")
print("=" * 60)
