"""L4-Day3: RAG API + Function Calling"""

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

# ── 嵌入 ──
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

def embed_texts(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        pooled = model(**inputs).last_hidden_state.mean(dim=1)
    return (pooled / torch.norm(pooled, dim=1, keepdim=True)).numpy()

class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.vectors = []

    def add_documents(self, documents):
        self.documents = documents
        self.vectors = embed_texts([d.page_content for d in documents])

    def similarity_search(self, query, k=2):
        q_vec = embed_texts([query])[0]
        scores = np.dot(self.vectors, q_vec)
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k]

vectorstore = SimpleVectorStore()

# 初始化知识库
docs = [
    Document("Python was created by Guido van Rossum in 1991."),
    Document("PyTorch was developed by Meta AI. It provides GPU-accelerated tensor computation."),
    Document("The Transformer architecture was introduced by Google in 2017."),
    Document("RAG combines a retriever with a generator."),
    Document("Chroma is an open-source vector database for AI applications."),
    Document("LangChain is a framework for developing LLM applications with chains and agents."),
]
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
vectorstore.add_documents(splitter.split_documents(docs))
print(f"▶ 知识库：{len(docs)} 篇 → {len(vectorstore.vectors[0])} 维")

# ── LLM ──
client = httpx.Client(timeout=30)

def call_llm(messages, tools=None):
    body = {
        "model": "deepseek-v4-flash", "messages": messages,
        "temperature": 0.3, "thinking": {"type": "disabled"}, "stream": False,
    }
    if tools: body["tools"] = tools
    r = client.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

# ── 工具 ──
TOOLS = [{"type": "function", "function": {
    "name": "search_knowledge",
    "description": "Search the knowledge base for relevant information about Python, PyTorch, Transformer, RAG, Chroma, LangChain",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}}]
TOOL_IMPLS = {"search_knowledge": lambda query: "\n".join(d.page_content for d in vectorstore.similarity_search(query))}

def rag_with_fc(query: str, max_rounds=5) -> str:
    msgs = [
        {"role": "system", "content": "Use search_knowledge to look up information before answering technical questions. For greetings or casual chat, answer directly."},
        {"role": "user", "content": query},
    ]
    for _ in range(max_rounds):
        msg = call_llm(msgs, tools=TOOLS)
        if not msg.get("tool_calls"):
            return msg["content"]
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
        for tc in msg["tool_calls"]:
            result = TOOL_IMPLS.get(tc["function"]["name"], lambda: "unknown")(**json.loads(tc["function"]["arguments"] or "{}"))
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    return msgs[-1].get("content", "")

# ── FastAPI ──
app = FastAPI(title="RAG API (FC)", version="1.0.0")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    return {"answer": rag_with_fc(req.question)}

@app.get("/health")
def health():
    return {"status": "ok", "docs_count": len(vectorstore.documents)}

if __name__ == "__main__":
    import uvicorn
    print("启动 RAG API (FC 版)...")
    print("  POST /query  →  Function Calling + 向量检索")
    print("  GET  /health →  健康检查")
    uvicorn.run(app, host="0.0.0.0", port=8000)
