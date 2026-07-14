"""RAG API + Function Calling（多工具版）

工具：
  - search_knowledge : 向量检索知识库
  - add_document     : 动态添加新知识（自动分块嵌入）
  - summarize        : 文本摘要
  - translate        : 翻译

API 端点：
  POST /query   → FC 驱动问答
  POST /doc     → 批量加文档
  GET  /health  → 健康检查
"""

import sys, os, json

_REAL_USER_SITE = r"C:\Users\inervers\AppData\Roaming\Python\Python313\site-packages"
if os.path.isdir(_REAL_USER_SITE) and _REAL_USER_SITE not in sys.path:
    sys.path.insert(0, _REAL_USER_SITE)

os.environ.setdefault("HF_HOME", r"C:\Users\inervers\Desktop\OH-WorkSpace\dl-learning\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import numpy as np
import httpx
from transformers import AutoTokenizer, AutoModel
import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("请设置环境变量 DEEPSEEK_API_KEY")
    exit(1)

# =============================================
# 嵌入模型
# =============================================
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

def embed_texts(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        pooled = model(**inputs).last_hidden_state.mean(dim=1)
    return (pooled / torch.norm(pooled, dim=1, keepdim=True)).numpy()

# =============================================
# 向量存储（支持增量添加）
# =============================================
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

class VectorStore:
    def __init__(self):
        self.documents = []   # list[Document]
        self.vectors = np.empty((0, 384))  # MiniLM 输出 384 维

    def add_documents(self, documents):
        """批量添加文档（初始化用）"""
        self.documents = documents
        if documents:
            self.vectors = embed_texts([d.page_content for d in documents])

    def append_document(self, document):
        """追加单篇文档"""
        self.documents.append(document)
        vec = embed_texts([document.page_content])
        self.vectors = np.vstack([self.vectors, vec])

    def similarity_search(self, query, k=3):
        if not self.documents:
            return []
        q_vec = embed_texts([query])[0]
        scores = np.dot(self.vectors, q_vec)
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k]

vectorstore = VectorStore()

# 初始化知识库
init_docs = [
    Document("Python was created by Guido van Rossum in 1991."),
    Document("PyTorch was developed by Meta AI. It provides GPU-accelerated tensor computation."),
    Document("The Transformer architecture was introduced by Google in 2017."),
    Document("RAG combines a retriever with a generator."),
    Document("Chroma is an open-source vector database for AI applications."),
    Document("LangChain is a framework for developing LLM applications with chains and agents."),
]
vectorstore.add_documents(splitter.split_documents(init_docs))
print(f"▶ 知识库初始加载：{len(vectorstore.documents)} 个块，{vectorstore.vectors.shape[1]} 维")

# =============================================
# DeepSeek LLM
# =============================================
client = httpx.Client(timeout=30)

def call_llm(messages, tools=None):
    body = {
        "model": "deepseek-v4-flash", "messages": messages,
        "temperature": 0.3, "thinking": {"type": "disabled"}, "stream": False,
    }
    if tools:
        body["tools"] = tools
    r = client.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

def _deepseek_ask(system: str, user: str) -> str:
    """简化的 DeepSeek 单轮调用，供工具内部复用"""
    body = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    r = client.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# =============================================
# 工具定义
# =============================================

TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "搜索知识库（向量语义检索），查找与问题相关的文档。适合 Python、PyTorch、Transformer、RAG、Chromadb、LangChain 等主题",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "add_document",
        "description": "向知识库添加一条新知识，之后被 search_knowledge 可检索到",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "知识标题"},
                "content": {"type": "string", "description": "知识正文"},
            },
            "required": ["title", "content"],
        },
    }},
    {"type": "function", "function": {
        "name": "summarize",
        "description": "对一段文本进行摘要总结。当用户要求总结/摘要一段内容时，请调用此工具，不要自己写总结",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "translate",
        "description": "将文本翻译为目标语言。当用户要求翻译时，请调用此工具，不要自己翻译",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_language": {"type": "string", "description": "目标语言，如 Chinese, English, Japanese, French"},
            },
            "required": ["text", "target_language"],
        },
    }},
]

# =============================================
# 工具实现
# =============================================

def _tool_search_knowledge(query: str) -> str:
    docs = vectorstore.similarity_search(query)
    if not docs:
        return "知识库为空"
    return "\n".join(d.page_content for d in docs)

def _tool_add_document(title: str, content: str) -> str:
    full_text = f"{title}：{content}"
    chunks = splitter.split_documents([Document(full_text)])
    for chunk in chunks:
        vectorstore.append_document(chunk)
    return f"成功添加文档「{title}」（{len(chunks)} 个分块），知识库共 {len(vectorstore.documents)} 个块"

def _tool_summarize(text: str) -> str:
    return _deepseek_ask(
        "You are a summarizer. Provide a concise summary in the same language as the input text.",
        f"Summarize this:\n\n{text}",
    )

def _tool_translate(text: str, target_language: str) -> str:
    return _deepseek_ask(
        f"You are a professional translator. Translate to {target_language}. Output only the translation.",
        text,
    )

TOOL_IMPLS = {
    "search_knowledge": _tool_search_knowledge,
    "add_document": _tool_add_document,
    "summarize": _tool_summarize,
    "translate": _tool_translate,
}

# =============================================
# FC 循环
# =============================================

SYSTEM_PROMPT = (
    "You are an AI assistant with dedicated tools for specific tasks.\n"
    "Available tools:\n"
    "  search_knowledge : search the vector knowledge base\n"
    "  add_document     : add new knowledge to the vector knowledge base\n"
    "  summarize        : summarize any text (use this when user asks to summarize)\n"
    "  translate        : translate text (use this when user asks to translate)\n\n"
    "Rules:\n"
    "1. FOR TECHNICAL QUESTIONS, ALWAYS use search_knowledge first.\n"
    "2. For greetings or casual chat, answer directly without tools.\n"
    "3. When the user asks to SUMMARIZE something, call the summarize tool.\n"
    "4. When the user asks to TRANSLATE something, call the translate tool.\n"
    "5. You CAN chain multiple tools: search → summarize → translate.\n"
    "6. After adding a document, verify with search_knowledge.\n"
    "Answer in the same language as the user."
)

def rag_with_fc(query: str, max_rounds=8) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    for _ in range(max_rounds):
        msg = call_llm(msgs, tools=TOOLS)
        if not msg.get("tool_calls"):
            return msg["content"]
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
        for tc in msg["tool_calls"]:
            fname = tc["function"]["name"]
            fargs = json.loads(tc["function"]["arguments"] or "{}")
            impl = TOOL_IMPLS.get(fname)
            result = impl(**fargs) if impl else f"未知工具：{fname}"
            print(f"  🛠  {fname}({json.dumps(fargs, ensure_ascii=False)})")
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    return msgs[-1].get("content", "")

# =============================================
# FastAPI 端点
# =============================================

app = FastAPI(title="RAG Agent API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

class DocRequest(BaseModel):
    title: str
    content: str

class DocResponse(BaseModel):
    message: str
    total_chunks: int

@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    return {"answer": rag_with_fc(req.question)}

@app.post("/doc")
def add_doc(req: DocRequest):
    """通过 API 添加知识到知识库"""
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(400, "标题和内容不能为空")
    result = _tool_add_document(req.title, req.content)
    return {"message": result, "total_chunks": len(vectorstore.documents)}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(vectorstore.documents),
        "tools": list(TOOL_IMPLS.keys()),
    }

if __name__ == "__main__":
    import uvicorn
    print("启动 RAG Agent API（多工具 FC 版）...")
    print("  POST /query  →  Function Calling 问答（search / add / summarize / translate）")
    print("  POST /doc    →  动态添加知识")
    print("  GET  /health →  健康检查")
    uvicorn.run(app, host="0.0.0.0", port=8000)
