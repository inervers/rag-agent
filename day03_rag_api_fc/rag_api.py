"""RAG API + Function Calling（文件持久化版）

工具：search_knowledge / add_document / summarize / translate
端点：POST /query  POST /doc  GET /health

数据持久化：向量存储在 vectordb/ 目录，重启不丢。
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
# 文件持久化向量存储
# =============================================
NP_DATA_DIR = os.path.join(os.path.dirname(__file__), "vectordb")
os.makedirs(NP_DATA_DIR, exist_ok=True)
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)


class PersistentVectorStore:
    """轻量文件持久化向量存储。

    原理与 Chroma 相同：
      - 向量存 .npy 文件
      - 文档内容存 .json 文件
      - 重启时自动从磁盘加载
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.vectors_path = os.path.join(data_dir, "vectors.npy")
        self.docs_path = os.path.join(data_dir, "documents.json")
        self.documents: list[str] = []
        self.vectors = np.empty((0, 384), dtype=np.float32)
        self._load()

    def _load(self):
        if os.path.isfile(self.docs_path) and os.path.isfile(self.vectors_path):
            self.documents = json.load(open(self.docs_path, "r", encoding="utf-8"))
            self.vectors = np.load(self.vectors_path)
            print(f"  加载持久化知识库：{len(self.documents)} 个块")

    def _save(self):
        json.dump(self.documents, open(self.docs_path, "w", encoding="utf-8"), ensure_ascii=False)
        np.save(self.vectors_path, self.vectors)

    def initialize(self, texts: list[str]):
        """首次初始化（仅当存储为空时）"""
        if len(self.documents) > 0:
            return False
        self.documents = texts
        self.vectors = embed_texts(texts)
        self._save()
        return True

    def append(self, text: str):
        """追加单条文档"""
        self.documents.append(text)
        vec = embed_texts([text])
        self.vectors = np.vstack([self.vectors, vec])
        self._save()

    def search(self, query: str, k=3) -> list[str]:
        if not self.documents:
            return []
        q_vec = embed_texts([query])[0]
        scores = np.dot(self.vectors, q_vec)
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k]

    @property
    def count(self) -> int:
        return len(self.documents)


vstore = PersistentVectorStore(NP_DATA_DIR)

# 首次启动时加载初始知识库
if vstore.count == 0:
    init_texts = [
        "Python was created by Guido van Rossum and first released in 1991. It is a high-level general-purpose programming language emphasizing code readability with significant indentation. Python supports multiple programming paradigms including structured, object-oriented, and functional programming. It has a large standard library and a vibrant ecosystem of third-party packages for web development, data science, machine learning, automation, and scientific computing.",
        "PyTorch was developed by Meta AI (Facebook AI Research) and released in 2016. It is an open-source machine learning framework that accelerates the path from research prototyping to production deployment. Key features include dynamic computation graphs (eager execution), GPU-accelerated tensor computation, automatic differentiation with Autograd, and a rich ecosystem including TorchVision, TorchText, and TorchAudio.",
        "The Transformer architecture was introduced by Google in the 2017 paper 'Attention Is All You Need'. Unlike RNNs or CNNs, it relies entirely on self-attention mechanisms to process sequential data. The architecture consists of an encoder and decoder, each built from stacked layers of multi-head self-attention and feed-forward networks. It has become the foundation for modern NLP models like BERT, GPT, T5, and has been extended to computer vision (ViT) and other domains.",
        "RAG (Retrieval-Augmented Generation) is a technique that combines a retriever and a generator. The retriever searches a knowledge base (like a vector database) for documents relevant to the user's question. These retrieved documents are then fed as context to the generator (an LLM) to produce an informed answer grounded in real sources. RAG reduces hallucination, enables knowledge updates without retraining, and allows citation of sources.",
        "Chroma is an open-source vector database built specifically for AI applications. It provides efficient storage and retrieval of embeddings with support for cosine similarity, L2 distance, and inner product. Chroma supports persistent storage, metadata filtering, and can be used as a drop-in replacement for simple in-memory vector stores. It integrates natively with LangChain and LlamaIndex.",
        "LangChain is an open-source framework designed to simplify the development of LLM applications. It provides modular abstractions for models, prompts, chains, memory, agents, and retrieval. LangChain supports LCEL (LangChain Expression Language) for composing pipelines with the pipe operator, integrated tool calling for agents, and native integration with vector stores, document loaders, and embedding models.",
    ]
    chunks = splitter.split_documents([Document(t) for t in init_texts])
    texts = [c.page_content for c in chunks]
    vstore.initialize(texts)
    print(f"▶ 知识库初始化：{vstore.count} 个块，持久化于 {NP_DATA_DIR}")
else:
    print(f"▶ 知识库已从磁盘加载：{vstore.count} 个块，{NP_DATA_DIR}")

# =============================================
# DeepSeek LLM
# =============================================
llm_client = httpx.Client(timeout=30)

def call_llm(messages, tools=None):
    body = {
        "model": "deepseek-v4-flash", "messages": messages,
        "temperature": 0.3, "thinking": {"type": "disabled"}, "stream": False,
    }
    if tools:
        body["tools"] = tools
    r = llm_client.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def _deepseek_ask(system: str, user: str) -> str:
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
    r = llm_client.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# =============================================
# 工具定义
# =============================================

TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "搜索知识库（向量语义检索），查找与问题相关的文档。适合 Python、PyTorch、Transformer、RAG、Chroma、LangChain 等主题",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "add_document",
        "description": "向知识库添加一条新知识，之后被 search_knowledge 可检索到。知识重启后不丢失",
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
        "description": "对一段文本进行摘要总结。当用户要求总结时请调用此工具，不要自己写总结",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "translate",
        "description": "将文本翻译为目标语言。当用户要求翻译时请调用此工具，不要自己翻译",
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
    docs = vstore.search(query)
    if not docs:
        return "知识库中未找到相关信息"
    return "\n".join(docs)

def _tool_add_document(title: str, content: str) -> str:
    full_text = f"{title}：{content}"
    chunks = splitter.split_documents([Document(full_text)])
    for chunk in chunks:
        vstore.append(chunk.page_content)
    return f"成功添加文档「{title}」（{len(chunks)} 个分块），知识库共 {vstore.count} 个块"

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
    "  summarize        : summarize any text\n"
    "  translate        : translate text to another language\n\n"
    "Rules:\n"
    "1. FOR TECHNICAL QUESTIONS, ALWAYS use search_knowledge first.\n"
    "2. For greetings or casual chat, answer directly without tools.\n"
    "3. When asked to SUMMARIZE something, you MUST call the summarize tool.\n"
    "4. When asked to TRANSLATE something, you MUST call the translate tool.\n"
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

app = FastAPI(title="RAG Agent API (Persistent)", version="1.1.0")


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
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(400, "标题和内容不能为空")
    result = _tool_add_document(req.title, req.content)
    return {"message": result, "total_chunks": vstore.count}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": vstore.count,
        "tools": list(TOOL_IMPLS.keys()),
        "storage": "file_persistent",
        "db_path": NP_DATA_DIR,
    }

if __name__ == "__main__":
    import uvicorn
    print(f"启动 RAG Agent API（文件持久化）...")
    print(f"  知识库：{vstore.count} 个块 → {NP_DATA_DIR}")
    print("  POST /query  →  Function Calling 问答")
    print("  POST /doc    →  动态添加知识（重启不丢）")
    print("  GET  /health →  健康检查")
    uvicorn.run(app, host="0.0.0.0", port=8000)
